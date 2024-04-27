@echo off
set /p commit_msg="Enter your commit message: "
set datetime=%date% %time%

:: Add all changed files to the staging area
git add *

:: Commit the changes with a timestamp
git commit -m "%commit_msg% - %datetime%"

:: Push the changes to the remote repository
git push origin master

echo Commit and push completed.
pause
