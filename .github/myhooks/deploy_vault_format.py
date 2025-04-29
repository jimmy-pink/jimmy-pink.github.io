import argparse
import os

import convert_obsidian_md
import article_index
import sidebar_generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Obsidian links to standard markdown links')
    parser.add_argument('--dir', default='.', help='Root directory to scan (default: current directory)')
    args = parser.parse_args()
    print(args.dir)

    base_dir = args.dir
    generator = sidebar_generator.SidebarGenerator(base_dir)
    generator.save()
    print("已生成sidebar和navbar")

    generator = article_index.ArticleIndexGenerator(base_dir)
    index_path = os.path.join(base_dir, '.obsidian', 'filename_index.json')
    index = generator.save_index(index_path, with_extension=True)

    # Get all files
    print(f"生成索引,文件路径： {index} ")

    convert_obsidian_md.main()
