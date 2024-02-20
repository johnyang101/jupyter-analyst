class Cell:
    
    def __init__(self, cell_type, content):
        self.cell_type = cell_type
        self.content = content
class CodeCell(Cell):
    
    def __init__(self, content, source='set_next_input', replace='False'):
        super().__init__('code', content)
        
        self.content = self.parse(content)
        self.source = source
        self.replace = replace
        
    def parse(self, content):
        lines = content.split('\n')
        parsed_lines = [line.strip() for line in lines if line.strip() != '']
        return '\n'.join(parsed_lines)
    
class MarkdownCell(Cell):

    def __init__(self, content):
        super().__init__('md', content)

hardcoded_outputs = [
    [CodeCell("""import scperturb
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import os""")],
    [CodeCell("""path = './AissaBenevolenskaya2021.h5ad'
adata = sc.read_h5ad(path)
adata.obs['perturbation'].value_counts()""")],
    [CodeCell("""import matplotlib.pyplot as plt
import seaborn as sns
# Identify mitochondrial genes. Adjust the prefix if necessary.
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # For human genes


# Calculate the percentage of mitochondrial gene expression for each cell
adata.obs['percent_mt'] = np.sum(
   adata[:, adata.var['mt']].X, axis=1).A1 / np.sum(adata.X, axis=1).A1 * 100




# Visualize the distribution of mitochondrial gene expression percentage
sns.histplot(adata.obs['percent_mt'], bins=50, kde=True)
plt.xlabel('Percentage of Mitochondrial Gene Expression')
plt.ylabel('Number of Cells')
plt.title('Mitochondrial Gene Expression Distribution')
plt.show()""")],
    [CodeCell("""# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Threshold set
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]
# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Example threshold
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]""")],
    [CodeCell("""# Normalize the data to make the total counts equal across cells
sc.pp.normalize_total(adata, target_sum=1e4)


# Logarithmize the data
sc.pp.log1p(adata)


# Identify highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)


# Keep only the highly variable genes
adata = adata[:, adata.var.highly_variable]
# Scale the data to zero mean and unit variance
sc.pp.scale(adata, max_value=10)


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)


sc.tl.leiden(adata, resolution=1.0)


sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='leiden')
sc.pl.pca_variance_ratio(adata, log=True)
sc.pl.pca(adata, color='perturbation')""")],
    [CodeCell("""# Run UMAP
sc.tl.umap(adata)


# Plot UMAP with clusters
sc.pl.umap(adata, color='leiden', title='UMAP projection of the data', palette='Set1')


# Plot UMAP with conditions
sc.pl.umap(adata, color='perturbation', title='UMAP projection by Condition', palette='Set2')""")],
    [CodeCell("""subset_adata = adata[adata.obs['ncounts'] > 9]


# Identify differentially expressed genes
sc.tl.rank_genes_groups(subset_adata, groupby='perturbation', method='wilcoxon', reference='control')


# Print results
sc.pl.rank_genes_groups(subset_adata)""")],
    [CodeCell("""# print 20 most expressed genes in each condition
top_genes_dict = {}


# Assuming subset_adata is already defined and contains the necessary data
for condition in subset_adata.obs['perturbation'].cat.categories[1:]:
   top_genes = subset_adata.uns['rank_genes_groups']['names'][condition][:20]
   top_genes_dict[condition] = top_genes


for condition, value in top_genes_dict.items():
   print(f"Condition: {condition}")
   print("\n".join(value))
   print("\n---\n")
# List of condition names to make it easier to access them by index or name
condition_names = list(top_genes_dict.keys())


# Ensure there are at least three conditions present
if len(condition_names) >= 3:
   # Convert the gene lists to sets for set operations
   genes_first_condition = set(top_genes_dict[condition_names[0]])
   genes_third_condition = set(top_genes_dict[condition_names[2]])


   # Find shared genes between the first and third conditions
   shared_genes_first_third = genes_first_condition.intersection(genes_third_condition)


   # Assuming we want to compare the second (index 1) and third (index 2) condition's gene lists
   genes_second_condition = set(top_genes_dict[condition_names[1]])
   # Find shared genes between the second and third conditions
   shared_genes_second_third = genes_second_condition.intersection(genes_third_condition)


   shared_genes_all = shared_genes_second_third.intersection(genes_first_condition)


   # Print the results
   print("Shared genes between the first and third lists:")
   print(shared_genes_first_third)
   print("\nShared genes between the second and third lists:")
   print(shared_genes_second_third)
   print("\nShared genes between all lists:")
   print(shared_genes_all)


else:
   print("Insufficient conditions available for comparison.")""")],
    [CodeCell("""# loading in gene name annotations
file_path = 'gene_name_annotations.txt'
df_annotation = pd.read_csv(file_path, sep='\t')
df_annotation.head()""")],
    [CodeCell("""# Open a text file to save the output
with open('top_20_genes.txt', 'w') as outfile:
   def custom_print(*args, **kwargs):
       # Print to console
       print(*args, **kwargs)
       # Print to file
       print(*args, **kwargs, file=outfile)
  
   # Assuming subset_adata is already defined and contains the necessary data
   for condition in subset_adata.obs['perturbation'].cat.categories[1:]:
       custom_print(f"Condition: {condition}")
       top_genes = subset_adata.uns['rank_genes_groups']['names'][condition][:20]
      
       for gene in top_genes:
           approved_name = df_annotation.loc[df_annotation['Approved symbol'] == gene, 'Approved name'].values
           if len(approved_name) > 0:
               custom_print(f"{gene} (Approved Name: {approved_name[0]})")
           else:
               custom_print(f"{gene} (No approved name found)")
      
       custom_print("\n---\n")


# Open a text file to save the output
with open('shared_genes_named.txt', 'w') as outfile:
   def custom_print(*args, **kwargs):
       # Print to console
       print(*args, **kwargs)
       # Print to file
       print(*args, **kwargs, file=outfile)


   # Assuming top_genes_dict and df_annotation are already defined and populated
   # List of condition names for easy access
   condition_names = list(top_genes_dict.keys())


   if len(condition_names) >= 3:
       # Shared genes between the first and third lists
       shared_genes_first_third = set(top_genes_dict[condition_names[0]]).intersection(set(top_genes_dict[condition_names[2]]))
       # Shared genes between the second and third lists
       shared_genes_second_third = set(top_genes_dict[condition_names[1]]).intersection(set(top_genes_dict[condition_names[2]]))
       # Optional: Shared genes across all three conditions
       shared_genes_all = shared_genes_second_third.intersection(top_genes_dict[condition_names[0]])


       # Output shared genes with annotations
       for shared_set, title in zip([shared_genes_first_third, shared_genes_second_third, shared_genes_all],
                                    ["Shared genes between the first and third lists:",
                                     "Shared genes between the second and third lists:",
                                     "Shared genes between all lists:"]):
           custom_print(title)
           for gene in shared_set:
               approved_name = df_annotation.loc[df_annotation['Approved symbol'] == gene, 'Approved name'].values
               if len(approved_name) > 0:
                   custom_print(f"{gene} (Approved Name: {approved_name[0]})")
               else:
                   custom_print(f"{gene} (No approved name found)")
           custom_print("\n---\n")
   else:
       custom_print("Insufficient conditions available for comparison.")""")],
    [CodeCell("""# loading in gene name annotations
file_path = 'gene_name_annotations.txt'
df_annotation = pd.read_csv(file_path, sep='\t')
df_annotation.head()""")],
    
]