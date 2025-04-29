import json
import os
from pathlib import Path


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


