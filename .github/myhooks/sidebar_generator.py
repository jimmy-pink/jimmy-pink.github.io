import argparse
import os
import re
import urllib.parse
from pathlib import Path
import json


class SidebarGenerator:
    def __init__(self, root_dir, sidebar_file="_sidebar.md", navbar_file="_navbar.md", exclude_paths=None, base_url=""):
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
        self.navbar_file = os.path.join(root_dir, navbar_file)
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

    def translate_dir_name(self, dir_name):
        """Translate directory name using the translation dictionary."""
        for k, v in self.term_trans_dict.items():
            if k in dir_name:
                return dir_name.replace(k, v)
        return dir_name

    def find_first_md_file(self, directory):
        """Find the first .md file in a directory, excluding excluded files."""
        try:
            files = [f for f in os.listdir(directory) 
                    if f.endswith('.md') 
                    and f not in self.exclude_file
                    and os.path.isfile(os.path.join(directory, f))]
            if files:
                return sorted(files, key=self.natural_sort_key)[0]
        except Exception:
            pass
        return None

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
        navbar_lines = []

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
            cn_dir_name = self.translate_dir_name(entry)
            sidebar_lines.append(f"{indent}* {cn_dir_name}")
            
            # Find first .md file in the directory for navbar link
            first_md = self.find_first_md_file(full_path)
            if first_md:
                md_rel_path = os.path.relpath(os.path.join(full_path, first_md), self.root_dir)
                encoded_path = urllib.parse.quote(md_rel_path)
                navbar_lines.append(f"  * [{cn_dir_name}]({self.base_url}{encoded_path})")
            else:
                # If no .md file found, just add the directory name without link
                navbar_lines.append(f"  * {cn_dir_name}")
            
            # Recursively scan subdirectory
            sub_sidebar, sub_navbar = self.scan_directory(full_path, level + 1)
            sidebar_lines.extend(sub_sidebar)
            navbar_lines.extend(sub_navbar)

        return sidebar_lines, navbar_lines

    def generate(self):
        """Generate the complete sidebar content."""
        # Start with homepage entry
        sidebar_md = ["* [首页](README.md)"]
        navbar_md = ["* [首页](README.md)", "* 学习阶段"]
        sidebar, navbar = self.scan_directory(self.root_dir)
        # Add all other entries
        sidebar_md.extend(sidebar)
        navbar_md.extend(navbar)
        navbar_md.append('* [GitHub](https://github.com/jimmy-pink/jimmy-pink.github.io)')
        return "\n".join(sidebar_md), "\n".join(navbar_md)

    def save(self):
        """Generate and save the sidebar file."""
        sidebar, navbar = self.generate()

        with open(self.sidebar_file, 'w', encoding='utf-8') as f:
            f.write(sidebar)

        with open(self.navbar_file, 'w', encoding='utf-8') as f:
            f.write(navbar)

        return sidebar, navbar
