#!/usr/bin/env python3
import os
import re
import json
import argparse
from urllib.parse import quote


def load_filename_index(root_dir):
    index_path = os.path.join(root_dir, '.obsidian', 'filename_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def process_link(link, filename_index, root_path=''):
    # 分离文件名和锚点
    parts = link.split('#', 1)
    filename = parts[0]
    anchor = '#' + parts[1] if len(parts) > 1 else ''

    # 处理文件扩展名
    if not filename.endswith('.md') and not '.' in filename:
        filename += '.md'

    # 获取相对路径
    relative_path = filename_index.get(filename, '')

    # URL 编码文件名部分
    encoded_filename = quote(filename)

    # 按路径分隔符分开，统计层数
    levels = len(relative_path.split(os.sep))
    # 每一层对应一个 "../"
    root_path = '../' * (levels - 1)

    # 构建完整链接
    full_path = os.path.join(root_path, relative_path, encoded_filename)
    return full_path + anchor


def convert_obsidian_links(content, filename_index, root_path=''):
    # 处理图片和视频链接
    content = re.sub(
        r'!\[\[([^\]]+)\]\]',
        lambda m: (
            f'<img src="{process_link(m.group(1), filename_index, root_path)}" alt="{os.path.basename(m.group(1))}">'
            if m.group(1).lower().endswith('.webp')
            else f'![{os.path.basename(m.group(1))}]({process_link(m.group(1), filename_index, root_path)})'
        ),
        content
    )

    # 处理带自定义标题的链接和常规链接
    content = re.sub(
        r'\[\[([^|\]]+)(?:\|([^\]]+))?\]\]',
        lambda
            m: f'[{m.group(2) if m.group(2) else m.group(1)}]({process_link(m.group(1), filename_index, root_path)})',
        content
    )

    return content


def process_markdown_files(root_dir):
    filename_index = load_filename_index(root_dir)

    for root, _, files in os.walk(root_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, root_dir)
        root_path = '../' * (len(rel_path.split(os.sep)) - 1) if rel_path != '.' else ''

        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    new_content = convert_obsidian_links(content, filename_index, root_path)

                    if content != new_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert Obsidian links to standard markdown links')
    parser.add_argument('--dir', default='.', help='Root directory to scan (default: current directory)')
    args = parser.parse_args()

    process_markdown_files(args.dir)
