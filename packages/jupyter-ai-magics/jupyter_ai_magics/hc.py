hardcoded_outputs = [
    """;import scperturb\nimport scanpy as sc\nimport pandas as pd\nimport numpy as np\nimport anndata\nimport os""",
    """What data file should we use and describe the data?;""",
    """;path = './AissaBenevolenskaya2021.h5ad'\nadata = sc.read_h5ad(path)\nadata.obs['perturbation'].value_counts()""",
    """;import matplotlib.pyplot as plt\nimport seaborn as sns\n# Identify mitochondrial genes. Adjust the prefix if necessary.\nadata.var['mt'] = adata.var_names.str.startswith('MT-')  # For human genes\n\n\n# Calculate the percentage of mitochondrial gene expression for each cell\nadata.obs['percent_mt'] = np.sum(\nadata[:, adata.var['mt']].X, axis=1).A1 / np.sum(adata.X, axis=1).A1 * 100\n\n\n\n\n# Visualize the distribution of mitochondrial gene expression percentage\nsns.histplot(adata.obs['percent_mt'], bins=50, kde=True)\nplt.xlabel('Percentage of Mitochondrial Gene Expression')\nplt.ylabel('Number of Cells')\nplt.title('Mitochondrial Gene Expression Distribution')\nplt.show()"""
    """# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Threshold set
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]
# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Example threshold
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]""",
"""# Normalize the data to make the total counts equal across cells
sc.pp.normalize_total(adata, target_sum=1e4)


# Logarithmize the data
sc.pp.log1p(adata)


# Identify highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)


# Keep only the highly variable genes
adata = adata[:, adata.var.highly_variable]""",
"""# Scale the data to zero mean and unit variance
sc.pp.scale(adata, max_value=10)


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)


sc.tl.leiden(adata, resolution=1.0)


sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='leiden')
sc.pl.pca_variance_ratio(adata, log=True)
sc.pl.pca(adata, color='perturbation')""",
"""# Run UMAP
sc.tl.umap(adata)


# Plot UMAP with clusters
sc.pl.umap(adata, color='leiden', title='UMAP projection of the data', palette='Set1')


# Plot UMAP with conditions
sc.pl.umap(adata, color='perturbation', title='UMAP projection by Condition', palette='Set2')""",
"""subset_adata = adata[adata.obs['ncounts'] > 9]


# Identify differentially expressed genes
sc.tl.rank_genes_groups(subset_adata, groupby='perturbation', method='wilcoxon', reference='control')


# Print results
sc.pl.rank_genes_groups(subset_adata)""",
"""# print 20 most expressed genes in each condition
top_genes_dict = {}


# Assuming subset_adata is already defined and contains the necessary data
for condition in subset_adata.obs['perturbation'].cat.categories[1:]:
   top_genes = subset_adata.uns['rank_genes_groups']['names'][condition][:20]
   top_genes_dict[condition] = top_genes


for condition, value in top_genes_dict.items():
   print(f"Condition: {condition}")
   print("\n".join(value))
   print("\n---\n")""",
"""# List of condition names to make it easier to access them by index or name
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
   print("Insufficient conditions available for comparison.")""",
"""# loading in gene name annotations
file_path = 'gene_name_annotations.txt'
df_annotation = pd.read_csv(file_path, sep='\t')
df_annotation.head()""",
"""# Open a text file to save the output
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
       custom_print("Insufficient conditions available for comparison.")""",
"""import gseapy as gp
# Assuming subset_adata is your AnnData object already filtered and with DEGs identified
# Let's define a cutoff for adjusted p-values (commonly used is 0.05)
pval_cutoff = 0.05


# Extract DEGs for a specific condition; you can loop through conditions as needed
degs = {}
for condition in subset_adata.uns['rank_genes_groups']['names'].dtype.names:
   # Get indices where the adjusted p-value is below the cutoff
   significant_genes_mask = subset_adata.uns['rank_genes_groups']['pvals_adj'][condition] < pval_cutoff
   # Extract gene names based on the mask
   significant_genes = subset_adata.uns['rank_genes_groups']['names'][condition][significant_genes_mask]
   degs[condition] = significant_genes.tolist()


# Loop through each condition to perform GSEA
for condition, gene_list in degs.items():
   if gene_list:  # Ensure there are genes to analyze
       # Perform Enrichment Analysis
       enr = gp.enrichr(gene_list=gene_list,
                        gene_sets=['GO_Biological_Process_2021'], 
                        organism='Human',  # Adjust as necessary
                        #description=condition,
                        outdir=os.curdir,  # Adjust or leave as None to not save
                        no_plot=False, 
                        cutoff=0.05  # Significance cutoff for the enrichment analysis itself
                        )


       # Display the top 10 enriched terms for the condition
       print(f"Condition: {condition}")
       print(enr.results.head(10))
       print("\n---\n")""",
"""MORF4L1 (Mortality factor 4 like 1) is a human gene that encodes the protein Mortality factor 4 like 1. This protein is characterized by a specific structural layout and functional domains, including a chromodomain (CD) at its NH2 terminus and an MRG domain at its carboxyl terminus, with a nuclear localization signal located between these two domains. The protein plays a crucial role in cell cycle progression and proliferation, and is involved in various physical interactions with other proteins and nucleic acids. Alterations in the MORF4L1 gene have been linked to various diseases, including coronary artery disease and several types of cancer.<HTML>""",
'''import requests
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt


def fetch_string_data(query_protein, species=9606, required_score=400):
   """
   Fetches interaction data for a given protein from STRING.
  
   Args:
   - query_protein (str): The protein name or STRING ID.
   - species (int): NCBI Taxonomy ID for the species (default is 9606 for Homo sapiens).
   - required_score (int): Minimum interaction score (default is 400).
  
   Returns:
   - A list of tuples representing the edges between interacting proteins.
   """
   STRING_API_URL = "https://string-db.org/api"
   output_format = "json"
   method = "network"
   request_url = f"{STRING_API_URL}/{output_format}/{method}?identifiers={query_protein}&species={species}&required_score={required_score}"


   response = requests.get(request_url)
   if response.status_code == 200:
       interactions = response.json()
       edges = [(interaction["preferredName_A"], interaction["preferredName_B"]) for interaction in interactions]
       return edges
   else:
       return []


def visualize_network(edges, file_name="protein_interactions.html"):
   """
   Saves a network of protein interactions as an HTML file.
  
   Args:
   - edges (list of tuple): Edges representing protein interactions.
   - file_name (str): Name of the file to save the HTML visualization.
   """
   G = nx.Graph()
   G.add_edges_from(edges)
  
   net = Network(notebook=False)  # Changed to False since we're not displaying in notebook
   net.from_nx(G)
   net.write_html(file_name)  # Saves the visualization as an HTML file


# Example usage
query_protein = "MORF4L1" 
edges = fetch_string_data(query_protein)
network = visualize_network(edges);''',
'''Retrieving related experiments …. Found 1 experimental analysis related to MORFL41. Would you like to load in the data from this experiment?''',
'''Loading ‘8h.h5ad’ from E-MTAB-6754 The experimental conditions were: cisplatin, cisplatin_olaparib, crizotinib, dabrafenib, dacarbazine, dasatinib, decitabine, dexamethasone, erlotinib, everolimus, hydroxyurea, imatinib, ixazomib, ixazomib_lenalidomide_dexamethasone, lenalidomide, melphalan, midostaurin, mln2480, olaparib, paclitaxel, palbociclib, panobinostat, pomalidomide_carfilzomib_dexamethasone, regorafenib, sorafenib, staurosporine, temozolomide, trametinib, trametinib_dabrafenib, trametinib_erlotinib, trametinib_midostaurin, trametinib_panobinostat, ulixertinib, vemurafenib_cobimetinib, vindesine.'''

]