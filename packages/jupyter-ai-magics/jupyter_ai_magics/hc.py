from IPython.display import HTML, JSON, Markdown, Math
import re
import base64
from xml.etree import ElementTree as ET
class TextWithMetadata:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata

    def __str__(self):
        return self.text

    def _repr_mimebundle_(self, include=None, exclude=None):
        return ({"text/plain": self.text}, self.metadata)


class Base64Image:
    def __init__(self, mimeData, metadata):
        mimeDataParts = mimeData.split(",")
        self.data = base64.b64decode(mimeDataParts[1])
        self.mimeType = re.sub(r";base64$", "", mimeDataParts[0])
        self.metadata = metadata

    def _repr_mimebundle_(self, include=None, exclude=None):
        return ({self.mimeType: self.data}, self.metadata)

DISPLAYS_BY_FORMAT = {
    "code": None,
    "html": HTML,
    "image": Base64Image,
    "markdown": Markdown,
    "math": Math,
    "md": Markdown,
    "json": JSON,
    "text": TextWithMetadata,
}

def parse_output(output):
    # Parse the XML-like string
    root = ET.fromstring(f"<root>{output}</root>")
    parsed_outputs = []
    
    # Iterate over the elements and create appropriate display objects
    for cell in root:
        if cell.tag == 'cell':
            for content in cell:
                display_class = DISPLAYS_BY_FORMAT.get(content.tag)
                if display_class:
                    # Create an instance of the display class with content text
                    instance = display_class(content.text, {})
                    parsed_outputs.append(instance)
    return parsed_outputs




hardcoded_outputs = [
   ['code', """;import scperturb\nimport scanpy as sc\nimport pandas as pd\nimport numpy as np\nimport anndata\nimport os"""],
    ['md', """What data file should we use and describe the data?"""],
    ['code', """;path = './AissaBenevolenskaya2021.h5ad'\nadata = sc.read_h5ad(path)\nadata.obs['perturbation'].value_counts()"""],
    ['md', 'What would you like to do next?'],
    ['code', """;import matplotlib.pyplot as plt\nimport seaborn as sns\n# Identify mitochondrial genes. Adjust the prefix if necessary.\nadata.var['mt'] = adata.var_names.str.startswith('MT-')  # For human genes\n\n\n# Calculate the percentage of mitochondrial gene expression for each cell\nadata.obs['percent_mt'] = np.sum(\nadata[:, adata.var['mt']].X, axis=1).A1 / np.sum(adata.X, axis=1).A1 * 100\n\n\n\n\n# Visualize the distribution of mitochondrial gene expression percentage\nsns.histplot(adata.obs['percent_mt'], bins=50, kde=True)\nplt.xlabel('Percentage of Mitochondrial Gene Expression')\nplt.ylabel('Number of Cells')\nplt.title('Mitochondrial Gene Expression Distribution')\nplt.show()"""
    """# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Threshold set
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]
# Filter out cells with high mitochondrial gene expression
# Adjust the threshold based on your data
threshold_mt = 10  # Example threshold
adata = adata[adata.obs['percent_mt'] < threshold_mt, :]"""],
['md', 'What would you like to do next?'], 
['code', """# Normalize the data to make the total counts equal across cells
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
sc.pl.pca(adata, color='perturbation')"""],
['md', 'What would you like to do next? '],
['code', """# Run UMAP
sc.tl.umap(adata)


# Plot UMAP with clusters
sc.pl.umap(adata, color='leiden', title='UMAP projection of the data', palette='Set1')


# Plot UMAP with conditions
sc.pl.umap(adata, color='perturbation', title='UMAP projection by Condition', palette='Set2')"""],
['code', """subset_adata = adata[adata.obs['ncounts'] > 9]


# Identify differentially expressed genes
sc.tl.rank_genes_groups(subset_adata, groupby='perturbation', method='wilcoxon', reference='control')


# Print results
sc.pl.rank_genes_groups(subset_adata)"""],
['md', 'What would you like to do next? '],
['code', """# print 20 most expressed genes in each condition
top_genes_dict = {}


# Assuming subset_adata is already defined and contains the necessary data
for condition in subset_adata.obs['perturbation'].cat.categories[1:]:
   top_genes = subset_adata.uns['rank_genes_groups']['names'][condition][:20]
   top_genes_dict[condition] = top_genes


for condition, value in top_genes_dict.items():
   print(f"Condition: {condition}")
   print("\n".join(value))
   print("\n---\n")"""],
['md', 'What would you like to do next? '],
['code', """# List of condition names to make it easier to access them by index or name
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
   print("Insufficient conditions available for comparison.")"""],
['md', 'What would you like to do next? '],
['code', """# loading in gene name annotations
file_path = 'gene_name_annotations.txt'
df_annotation = pd.read_csv(file_path, sep='\t')
df_annotation.head()"""],
['md', 'What would you like to do next? '],
['code', """# Open a text file to save the output
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
       custom_print("Insufficient conditions available for comparison.")"""],
['md', 'What would you like to do next? '],
['md', 'To perform gene set enrichment analysis you can use the gesapy library and the enrichr algorithm with “GO_Biological_Process_2021” as the gene set. Would you like to proceed?'],
['code',"""import gseapy as gp
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
       print("\n---\n")"""],
['md', 'What would you like to do next? '],
['md', "SOX4 (SRY-box 4) is a transcription factor within the SOX family crucial for various cellular mechanisms, including differentiation, apoptosis, and proliferation.\nIts gene expression is pivotal in normal developmental and homeostatic processes.\n\nIn oncology, SOX4 exhibits dual functionality, oscillating between oncogenic and tumor-suppressive roles contingent on the cancer type and cellular milieu.\nAs an oncogene, SOX4 overexpression has been linked to tumorigenesis in cancers like breast, prostate, lung, and leukemia, promoting cell proliferation, impeding apoptosis, and enhancing metastasis.\nSpecifically, its overexpression in breast cancer correlates with poor prognosis and therapy resistance.\n\nConversely, instances exist where SOX4 behaves as a tumor suppressor, though this is less frequently observed.\nIts involvement extends to cancer stem cell (CSC) regulation, where SOX4 supports CSC maintenance and survival, implicating it in tumor growth, recurrence, and treatment resistance.\nSOX4 also regulates epithelial-mesenchymal transition (EMT), a process vital for metastasis, by enabling epithelial cells to gain mesenchymal traits and migratory capabilities."],
['md', 'What would you like to do next? '],
['code', '''import requests
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
query_protein = "SOX4" 
edges = fetch_string_data(query_protein)
network = visualize_network(edges);'''],
['md', 'What would you like to do next? '],
['md', '''Retrieving related experiments …. Found 1 experiment and 1 paper related to SOX4 and EMT. The experiment PC9D3-ERL2 scRNA-seq has SOX4 in a text file of highly expressed genes in PC9 cells treated with 2 µM erlotinib. The paper “Single-cell transcriptional changes associated with drug tolerance and response to combination therapies in cancer” has the following info: 
"scRNA-seq analysis of tumor tissues showed that osimertinib treatment produced several novel cell populations compared to vehicle-treated animals (Supplementary Fig. 14c), with many of the top markers (GPRC5A, SOX4, FOS, JUN, IGFBP3, and ALDH3A1), gene signatures (Fig. 7a, b and Supplementary Fig. 14d–h) and LINCS small molecules (Supplementary Data 22) also observed in the PC9 cell culture model."
The first statement mentions SOX4 in the context of scRNA-seq analysis of tumor tissues following osimertinib treatment. It highlights that osimertinib treatment led to the emergence of novel cell populations, with SOX4 being one of the top markers identified along with other genes. This finding suggests that SOX4 may play a role in the response to osimertinib treatment and could be a potential marker for identifying specific cell populations in the context of drug tolerance and response to combination therapies in cancer
"EMT • Among the transcription factors that regulate EMT, SOX4, SLUG/SNAI2, and GATA6 were highly increased in DT cells."
The second statement discusses SOX4 in relation to the regulation of epithelial-mesenchymal transition (EMT) in the study. It notes that SOX4, along with other transcription factors, was highly increased in drug-tolerant cells. This indicates that SOX4 may be involved in driving EMT processes in cancer cells that exhibit drug tolerance, potentially contributing to their resistance to therapy. Understanding the role of SOX4 in EMT could provide insights into the molecular mechanisms underlying drug resistance and inform the development of targeted therapies to overcome resistance mechanisms'''
],
['md', 'What would you like to do next? '],
['md', '''Loading documents from experiment PC9D3-ERL2 scRNA-seq. 
Treatment protocol: 1x106 cells were seeded on a p100 plate. Next day, a respective drug was added to each plate containing 10 mL of media: 12.5 µL of 20 mM etoposide (prepared in DMSO) were added to obtain the final concentration of 25 µM; 10 μL of 2 mM erlotinib (prepared in DMSO) were used to obtain the final concentration of 2 µM; and for the drug combination, 10 μL of 1 mM erlotinib and crizotinib (both prepared in DMSO) were used to obtain the final concentration of 1 µM each. Medium was changed again next day, on new medium containing respective drug. Total treatment time was 3 days. Cells were trypsinized, centrifuged and resuspended in DMEM (no FBS), counted, and processed further for Drop-seq.
Growth protocol: Non-small cell lung carcinoma cell line PC9 (Sigma) was grown in RPMI, 5% \FBS (HyClone) media containing penicillin and streptomycin in 5% CO2 at 37ºC. Cells obtained from vendor were maintained for not more than 5 passages before collection for scRNA analysis.'''],
['md', 'What would you like to do next? '], 
['md', ' The previous experiment loaded in ‘hvg.h5ad’ and generated a violin plot with key ‘SOX4’ and grouped by ‘perturbation’. Loading in data and generating plot.' ], 
['code', """path = 'hvg.h5ad'
adata = sc.read_h5ad(path)


adata_sox4 = adata[adata[:, 'SOX4'].X > 0, :]


# Plot the violin plot for the filtered data
sc.pl.violin(adata_sox4, groupby='perturbation', keys='SOX4')"""],

]