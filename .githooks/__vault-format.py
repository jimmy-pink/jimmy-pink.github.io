import argparse
import os
import re
import urllib.parse
from pathlib import Path
import json
import convert_obsidian_md

class ArticleIndexGenerator:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.index = {}
        # 要排除的目录名
        self.exclude_dirs = {".git", ".idea", "test"}

    def generate_index(self, with_ext):
        """Generate index mapping filenames (with or without extension) to their relative paths"""
        for root, dirs, files in os.walk(self.root_path):
            # 过滤掉要排除的目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            rel_path = os.path.relpath(root, self.root_path)
            rel_path = "" if rel_path == "." else f"{rel_path}/"

            for file in files:
                filename = file if with_ext else os.path.splitext(file)[0]
                self.index[filename] = rel_path

        return self.index

    def save_index(self, output_file="index.json", with_extension=False):
        """Save the generated index to a JSON file"""
        if not self.index:
            self.generate_index(with_extension)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

        return output_file

    # 根据文件后缀名生成索引
    def filter_by_extension(self, extensions=None):
        """Filter index to only include files with specific extensions"""
        if not self.index:
            self.generate_index()

        if extensions is None:
            extensions = ['.md', '.ipynb', '.pdf', '.doc', '.numbers', '.png']

        filtered_index = {k: v for k, v in self.index.items()
                          if any(k.lower().endswith(ext.lower()) for ext in extensions)}

        return filtered_index


class SidebarGenerator:
    def __init__(self, root_dir, sidebar_file="_sidebar.md", exclude_paths=None, base_url=""):
        """
        Initialize the sidebar generator with configuration parameters.

        Args:
            root_dir: Root directory to scan for documentation files
            sidebar_file: Output sidebar file path
            exclude_paths: List of directories to exclude from scanning
            base_url: Base URL prefix for links
        """
        self.root_dir = root_dir
        self.sidebar_file = os.path.join(root_dir, sidebar_file)
        self.exclude_paths = exclude_paths or ['jupyter-notes', 'test', 'images', "IBM-AI-Engineer-Course/jupyter-demo"]
        self.exclude_file = ["README.md", "_sidebar.md", "_navbar.md"]
        # 词条翻译词典
        self.term_trans_dict = {
            "Preface": "前言",
            "EssentialBasics": "必备基础",
            "MachineLearning": "传统机器学习",
            "DeepLearning": "深度学习",
            "RAG": "RAG搜索增强",
            "-Full-Stack": "全栈",
            "RecommendSystem": "推荐系统",
            "Side-Project": "上手项目",
            "IBM-AI-Engineer-Course": "IBM全套AI工程师课程(Coursera)"
        }
        self.base_url = base_url

    def should_exclude(self, path):
        """Check if a path should be excluded from the sidebar."""
        return path.startswith(".") or path in self.exclude_paths

    def natural_sort_key(self, s):
        """
        Natural sort key function for sorting files and directories with numeric prefixes.
        This ensures "9-something" comes before "11-something".
        """
        # Extract and convert any numeric values to integers for proper numeric comparison
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    def scan_directory(self, directory, level=0):
        """
        Recursively scan a directory and generate sidebar entries.

        Args:
            directory: Directory to scan
            level: Current directory nesting level

        Returns:
            List of sidebar entry lines
        """
        sidebar_lines = []

        # Get all items in the current directory
        entries = os.listdir(directory)

        # Process files first (we want files to appear before subdirectories)
        markdown_files = [f for f in entries if f.endswith(".md") and f not in self.exclude_file
                          and os.path.isfile(os.path.join(directory, f))]

        for file in sorted(markdown_files, key=self.natural_sort_key):
            rel_path = os.path.relpath(os.path.join(directory, file), self.root_dir)
            name = os.path.splitext(file)[0]

            # Create properly indented entry with URL-encoded path
            indent = "  " * level
            encoded_path = urllib.parse.quote(rel_path)
            sidebar_lines.append(f"{indent}* [{name}]({self.base_url}{encoded_path})")

        # Process subdirectories
        subdirs = [d for d in entries if os.path.isdir(os.path.join(directory, d))]

        for entry in sorted(subdirs, key=self.natural_sort_key):
            full_path = os.path.join(directory, entry)
            rel_path = os.path.relpath(full_path, self.root_dir)

            if self.should_exclude(rel_path):
                continue

            # Add directory entry
            indent = "  " * level
            cn_dir_name = dir_name = os.path.basename(full_path)
            for k in self.term_trans_dict.keys():
                if str.__contains__(dir_name, k):
                    cn_dir_name = dir_name.replace(k, self.term_trans_dict[k])
            sidebar_lines.append(f"{indent}* {cn_dir_name}")

            # Recursively scan subdirectory
            sidebar_lines.extend(self.scan_directory(full_path, level + 1))

        return sidebar_lines

    def generate(self):
        """Generate the complete sidebar content."""
        # Start with homepage entry
        content = ["* [首页](README.md)"]

        # Add all other entries
        content.extend(self.scan_directory(self.root_dir))

        return "\n".join(content)

    def save(self):
        """Generate and save the sidebar file."""
        content = self.generate()

        with open(self.sidebar_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return content


if __name__ == "__main__":
    generator = SidebarGenerator("../")
    content = generator.save()
    # print(content)
    print(f"已生成: {generator.sidebar_file}")

    parser = argparse.ArgumentParser(description='Convert Obsidian links to standard markdown links')
    parser.add_argument('--dir', default='.', help='Root directory to scan (default: current directory)')
    args = parser.parse_args()

    print(args.dir)
    generator = ArticleIndexGenerator(args.dir)
    index_path = os.path.join(args.dir, '.githooks', 'filename_index.json')
    index = generator.save_index(index_path, with_extension=True)

    # Get all files
    print(f"生成索引,文件路径： {index} ")

    convert_obsidian_md.main()
