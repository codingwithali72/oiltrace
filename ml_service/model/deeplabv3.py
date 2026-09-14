import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model():
    """
    Creates a DeepLabV3+ model with a ResNet34 encoder.
    The model is configured to accept 2-channel SAR input (VV, VH)
    and output 1-channel raw logits (no sigmoid activation).
    """
    # Initialize the model with 3 input channels first to safely load the 
    # pretrained ImageNet weights for the encoder.
    # Note: We set activation=None to return raw logits.
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # -------------------------------------------------------------------------
    # ADAPTING THE FIRST CONVOLUTIONAL LAYER FOR 2-CHANNEL INPUT
    # -------------------------------------------------------------------------
    # The ResNet34 encoder's first convolutional layer (conv1) expects 3 input 
    # channels (RGB) due to ImageNet pretraining. Its weight tensor has shape:
    # [64, 3, 7, 7] (out_channels, in_channels, kernel_size, kernel_size).
    # 
    # To adapt this for our 2-channel SAR data (VV, VH) while preserving the 
    # benefits of the pretrained weights, we replace conv1 with a new Conv2d 
    # layer that accepts 2 input channels. 
    # 
    # We initialize the new 2-channel weights by taking the weights from the 
    # original R and G channels, and adding the weights of the discarded B 
    # channel to them equally (or alternatively, just keeping R and G). Here, 
    # we average the pre-trained weights across the 3 channels and duplicate 
    # them across our 2 new channels. This ensures that the overall magnitude 
    # of activations passing to the next layer remains roughly consistent with 
    # what the rest of the network expects.
    # -------------------------------------------------------------------------
    
    # Extract the original first conv layer
    original_conv1 = model.encoder.conv1
    
    # Create a new conv layer with 2 input channels, copying other parameters
    new_conv1 = nn.Conv2d(
        in_channels=2,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    
    with torch.no_grad():
        # Get original weights: shape [64, 3, 7, 7]
        orig_weights = original_conv1.weight.data
        
        # Strategy: Sum the original 3 channels and divide by 2, then copy 
        # to the new 2 channels. This preserves the total expected sum of 
        # the inputs to the feature maps.
        # Another common strategy is simply slicing: orig_weights[:, :2, :, :]
        # but summing/averaging distributes the learned filters better.
        
        # Calculate the mean across the input channel dimension (dim=1)
        # Resulting shape: [64, 1, 7, 7]
        mean_weights = orig_weights.mean(dim=1, keepdim=True)
        
        # Duplicate the mean weights across the 2 new input channels
        # Resulting shape: [64, 2, 7, 7]
        new_weights = mean_weights.repeat(1, 2, 1, 1)
        
        # Assign the calculated weights to the new layer
        new_conv1.weight.data = new_weights
        
        # Copy bias if it exists
        if original_conv1.bias is not None:
            new_conv1.bias.data = original_conv1.bias.data
            
    # Replace the original conv1 layer with our modified 2-channel layer
    model.encoder.conv1 = new_conv1
    
    return model

if __name__ == "__main__":
    # Smoke test to verify shapes
    model = create_model()
    print("Model successfully created.")
    
    # Create a dummy batch of 2 images, 2 channels, 512x512
    dummy_input = torch.randn(2, 2, 512, 512)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 1, 512, 512), "Output shape mismatch!"
    print("Smoke test passed: Output shape is correct (raw logits).")
