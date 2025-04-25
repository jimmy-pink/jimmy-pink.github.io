import os
import urllib.parse
import re


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
            dir_name = os.path.basename(full_path)
            sidebar_lines.append(f"{indent}* {dir_name}")

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

    print(content)
    print(f"已生成: {generator.sidebar_file}")