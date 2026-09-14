# OilTrace Build Progress

## Phase 0 — Project Skeleton Setup
- Status: DONE
- What was built: Initialized a git repository, created the required folder structure (/ml-service, /ship-detection, /ais-service, /backend, /frontend, /notebooks, /data, /docs) with .gitkeep placeholders, and set up a `.gitignore` specifically configured for a mixed Python, Java, and Node project. Created `README.md` and this `PROGRESS.md` file.
- Test performed: Ran `git add .`, `git status`, and `Get-ChildItem -Directory -Recurse | Select-Object FullName` to ensure folders were tracked properly and that the data folder is properly gitignored.
- Test result: 
```
warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'PROGRESS.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'SIH26143_Master_Build_Blueprint_Final.md', LF will be replaced by CRLF the next time Git touches it
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   .gitignore
	new file:   PROGRESS.md
	new file:   README.md
	new file:   SIH26143_Master_Build_Blueprint_Final.md
	new file:   ais-service/.gitkeep
	new file:   backend/.gitkeep
	new file:   docs/.gitkeep
	new file:   frontend/.gitkeep
	new file:   ml-service/.gitkeep
	new file:   notebooks/.gitkeep
	new file:   ship-detection/.gitkeep


FullName                    
--------                    
C:\BlueVector\ais-service   
C:\BlueVector\backend       
C:\BlueVector\data          
C:\BlueVector\docs          
C:\BlueVector\frontend      
C:\BlueVector\ml-service    
C:\BlueVector\notebooks     
C:\BlueVector\ship-detection
```
- Issues/notes: The `/data` directory is properly created, but intentionally missing from git staging because it is correctly ignored by the `/data/` rule in `.gitignore`.
