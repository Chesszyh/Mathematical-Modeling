@echo off
rem 创建一个结果文件
set outputFile=pdf_files.txt
rem 清空或新建结果文件
echo. > %outputFile%

rem 递归遍历当前目录及其子目录
for /r %%i in (*.pdf) do (
    echo %%~nxi >> %outputFile%
)

echo 所有PDF文件名已保存到 %outputFile%
pause
