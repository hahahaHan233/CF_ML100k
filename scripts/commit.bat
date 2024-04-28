@echo off
cd ..
set /p commit_message="Enter commit message: "
git add -A
git commit -m "%commit_message%"
git push
echo.
echo Commit and push finished.
pause
