#!/bin/bash

# 初始化脚本目录
SCRIPT_DIR="scripts/anomaly_detection/SMAP"
SELF_NAME=$(basename "$0")  # 当前脚本文件名

echo "开始批量添加执行权限（目录: $SCRIPT_DIR）..."

# 给目录及子目录下所有.sh文件添加执行权限
find "$SCRIPT_DIR" -type f -name "*.sh" -exec chmod +x {} \;

echo "完成！所有.sh文件已添加执行权限"

# 显示处理的文件列表
#echo "处理的文件列表："
#find "$SCRIPT_DIR" -type f -name "*.sh" -exec ls -la {} \;
#
## 顺序执行目录下的每个脚本，排除自己
#echo "开始顺序执行 $SCRIPT_DIR 中的脚本..."
#while IFS= read -r script; do
#    # 排除当前初始化脚本自己
#    if [ "$(basename "$script")" = "$SELF_NAME" ]; then
#        continue
#    fi
#
#    echo "正在执行 $script ..."
#    make run_module module="bash $script"
#done < <(find "$SCRIPT_DIR" -type f -name "*.sh" | sort)
#
#echo "所有脚本执行完成！"
