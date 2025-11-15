library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(32, "naive", 159.639);
df[2, ] <- c(64, "naive", 152.215);
df[3, ] <- c(128, "naive", 178.534);
df[4, ] <- c(256, "naive", 208.44);
df[5, ] <- c(512, "naive", 516.736);
df[6, ] <- c(1024, "naive", 2293.83);

df[7, ] <- c(32, "tiled", 163.185);
df[8, ] <- c(64, "tiled", 145.943);
df[9, ] <- c(128, "tiled", 153.848);
df[10, ] <- c(256, "tiled", 177.322);
df[11, ] <- c(512, "tiled", 414.876);
df[12, ] <- c(1024, "tiled", 1606.38);

df[13, ] <- c(32, "tiled+tensor", 137.928);
df[14, ] <- c(64, "tiled+tensor", 153.848);
df[15, ] <- c(128, "tiled+tensor", 162.935);
df[16, ] <- c(256, "tiled+tensor", 178.564);
df[17, ] <- c(512, "tiled+tensor", 359.723);
df[18, ] <- c(1024, "tiled+tensor", 893.532);

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
    xlab("N=M=R") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("Matrix Multiplication (MM)") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("mm_tensor.pdf", plot);
