import convert_obsidian_md
import deploy_vault_format

base_dir = "../../"
generator = deploy_vault_format.SidebarGenerator(base_dir)
content = generator.save()
# print(content)
print(f"已生成: {generator.sidebar_file}")

generator = deploy_vault_format.ArticleIndexGenerator(base_dir)
index = generator.save_index(base_dir + ".obsidian/filename_index.json", with_extension=True)

# Get all files
print(f"生成索引,文件路径： {index} ")
