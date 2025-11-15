library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(128, "naive", 188.634);
df[2, ] <- c(256, "naive", 252.614);
df[3, ] <- c(512, "naive", 553.387);
df[4, ] <- c(1024, "naive", 1685.99);
df[5, ] <- c(2048, "naive", 5977.25);
df[6, ] <- c(4096, "naive", 23253.6);

df[7, ] <- c(128, "tiled", 462.076);
df[8, ] <- c(256, "tiled", 182.251);
df[9, ] <- c(512, "tiled", 336.701);
df[10, ] <- c(1024, "tiled", 878.026);
df[11, ] <- c(2048, "tiled", 3321.99);
df[12, ] <- c(4096, "tiled", 13083.8);

df[13, ] <- c(128, "tiled+const", 898.554);
df[14, ] <- c(256, "tiled+const", 684.242);
df[15, ] <- c(512, "tiled+const", 164.579);
df[16, ] <- c(1024, "tiled+const", 459.401);
df[17, ] <- c(2048, "tiled+const", 1712.84);
df[18, ] <- c(4096, "tiled+const", 6634.13);

# set the names of the columns
colnames(df) <- c("size", "type", "latency");

# convert the columns in numeric data type
df$size <- as.numeric(as.character(df$size));
df$latency <- as.numeric(as.character(df$latency));

print(df)

# plotting
plot <- ggplot(df, aes(x = size, y = latency, color = type, group = type)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab("matrix size (L)") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("Convolution") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("convolution.pdf", plot);
