library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(64, "naive", 145.303);
df[2, ] <- c(128, "naive", 215.744);
df[3, ] <- c(256, "naive", 616.555);
df[4, ] <- c(512, "naive", 3300.19);
df[5, ] <- c(1024, "naive", 17472.4);

df[6, ] <- c(64, "tiled", 150.352);
df[7, ] <- c(128, "tiled", 252.663);
df[8, ] <- c(256, "tiled", 942.206);
df[9, ] <- c(512, "tiled", 6029.5);
df[10, ] <- c(1024, "tiled", 32122.8);

df[11, ] <- c(64, "coarsening", 184.416);
df[12, ] <- c(128, "coarsening", 245.19);
df[13, ] <- c(256, "coarsening", 634.058);
df[14, ] <- c(512, "coarsening", 3632.88);
df[15, ] <- c(1024, "coarsening", 18359);

df[16, ] <- c(64, "r_coarsening", 182.682);
df[17, ] <- c(128, "r_coarsening", 247.534);
df[18, ] <- c(256, "r_coarsening", 584.976);
df[19, ] <- c(512, "r_coarsening", 2972.73);
df[20, ] <- c(1024, "r_coarsening", 15860.4);

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
    xlab("input size (N)") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("7-point 3D stencil") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("stencil.pdf", plot);
