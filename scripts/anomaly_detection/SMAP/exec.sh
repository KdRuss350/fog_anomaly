#!/bin/bash

# 批量给所有.sh文件添加执行权限

echo "开始批量添加执行权限..."

# 给当前目录及子目录中所有.sh文件添加执行权限
find . -name "*.sh" -type f -exec chmod +x {} \;

echo "完成！所有.sh文件已添加执行权限"

# 显示处理结果
echo "处理的文件列表："
find . -name "*.sh" -type f -exec ls -la {} \;