# Trying to analyze published rnaseq data from github.com/stuberlab/Hashikawa-Hashikawa-2020
# This did not work because I was unable to find all of the necessary files

library(dplyr)
library(magrittr)
library("xlsx")
library(Seurat)

DATA_DIR <- "/storage/archive/sylwestrak/GSE137478_RAW/control"

cntl.data <- Read10X(DATA_DIR)
colnames(cntl.data) = paste0(colnames(cntl.data),"cntl")
cntl<- CreateSeuratObject(counts = cntl.data, min.cells = 3, min.features = 200, project = "10X_LHb")
cntl@meta.data$stim <- "cntl"

mito.features <- grep(pattern = "^mt-", x = rownames(x = cntl), value = TRUE)
percent.mito <- Matrix::colSums(x = GetAssayData(object = cntl, slot = 'counts')[mito.features, ]) / Matrix::colSums(x = GetAssayData(object = cntl, slot = 'counts'))
cntl[['percent.mito']] <- percent.mito

cntl <- subset(x = cntl, subset = nCount_RNA > 700 & nCount_RNA < 15000 & percent.mito < 0.20)

cntl<- NormalizeData(object = cntl,verbose = FALSE) 

cntl<- FindVariableFeatures(object =cntl,selection.method = "vst", nfeatures = 2000, verbose = FALSE)

cntl<- ScaleData(object = cntl, features = rownames(x =cntl), vars.to.regress = c("nCount_RNA", "percent.mito"))
cntl<- RunPCA(object = cntl, features = VariableFeatures(object =cntl), verbose = FALSE)
cntl <- JackStraw(object =cntl, num.replicate = 100)
cntl<- ScoreJackStraw(object = cntl, dims = 1:20)
cntl<- FindNeighbors(object =cntl, dims = 1:30)
cntl<- FindClusters(object = cntl, resolution = 0.8)

saveRDS(cntl,file = "/storage/archive/sylwestrak/cntl.rds")


counts<-as.matrix(cntl@assays$RNA@data)
write.table(data.frame("GENE"=rownames(counts),counts),file="/storage/archive/sylwestrak/counts.txt",row.names=FALSE,sep="\t")

markers <- FindAllMarkers(object = cntl, only.pos = TRUE, min.pct = 0.25)
top_50<-markers %>% group_by(cluster) %>% top_n(50)
write.table(data.frame("test"=as.character(rownames(top_50)),top_50),file="/storage/archive/sylwestrak/Top50Genes.txt",row.names=FALSE,col.names=c("",colnames(top_50)),sep="\t",eol="\n")



cluster<-Idents(object=cntl)
cluster<-as.matrix(cluster)
cluster[,1]<-as.character(cluster[,1])
cluster[,0]<-as.character(cluster[,0])
cluster<-data.frame("x"=rownames(cluster),cluster)
write.table(cluster,file="/storage/archive/sylwestrak/doublet/Cluster.txt",row.names=FALSE,col.names=c("","x"),sep="\t",eol="\n")


library("DoubletDecon")
location="/storage/archive/sylwestrak/doublet/" #Update as needed 
expressionFile=paste0(location, "counts.txt")
genesFile=paste0(location, "Top50Genes.txt")
clustersFile=paste0(location, "Cluster.txt")

newFiles=Seurat_Pre_Process(expressionFile, genesFile, clustersFile)

results=Main_Doublet_Decon(rawDataFile=newFiles$newExpressionFile, 
                           groupsFile=newFiles$newGroupsFile, 
                           filename="cntl", 
                           location=location,
                           fullDataFile=NULL, 
                           removeCC=FALSE, 
                           species="mmu", 
                           rhop=1.1, 
                           write=TRUE, 
                           PMF=TRUE, 
                           useFull=FALSE, 
                           heatmap=FALSE,
                           centroids=TRUE,
                           num_doubs=100, 
                           only50=TRUE,
                           min_uniq=4)

# The code seemed to work fine up to this point, but I was unable to find the files for 
# the mHb analysis. Also, I could not find any references to tyrosine hydroxylase in the data
