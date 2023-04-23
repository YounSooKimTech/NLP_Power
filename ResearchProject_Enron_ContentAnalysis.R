library(tidyverse)
library(tidyr)
library(ggplot2)
library(quanteda)
library(sentimentr)
library(quanteda.textmodels)

# read data
df = read.csv('https://raw.githubusercontent.com/YounSooKimTech/NLP_Power/main/Enron_merged_df.csv')
df = df %>% select(direction,Receiver_Email, Receiver_Position, Receiver_Rank, Sender_Email, Sender_Position, Sender_Rank, Subject, content)


# make a corpus
Enron_corpus = corpus(df, text_field = "content")
summary(docvars(Enron_corpus))



# top features (adjust n for uni, bi, tri)
n=1
Enron_dfm = tokens(Enron_corpus, remove_numbers = TRUE, 
                          remove_punct = TRUE, remove_url = TRUE,
                          remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  tokens_remove(pattern = "(?<=\\d{1,9})\\w+", valuetype = "regex") %>% 
  tokens_ngrams(n) %>% 
  dfm()
topfeatures(Enron_dfm, 20, groups = direction)

topfeatures(Enron_dfm, 100)

##############
###### Sentiment

## a dictionary that includes positive/negative word lists developed by Jockers (2017) and Hu & Liu (2004)**

## Load the dictionary:
setwd("C:/Users/younskim/Documents/TextAsData/W5_Sentiments")
base::load("sentiment_dictionary.rdata")

Enron_text = get_sentences(df$content)
Enron_sentiment = sentiment_by(Enron_text, by=df$direction)

Enron_sentiment
plot(Enron_sentiment)
ggplot(Enron_sentiment, aes(x=direction, y=ave_sentiment, fill=direction))+
  geom_bar(position="dodge",stat="identity") + theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme(legend.position = "none") + ggtitle("Sentiment by directions")



################
## topic modeling
##############

library(ggplot2)
library(topicmodels)
library(ldatuning)
library(quanteda)
library(stm)
library(dotwhisker)
library(tidytext)
library(tidyverse)

Enron_dfm = corpus(x= df, text_field = "content") %>%
  tokens(remove_numbers = TRUE, remove_punct = TRUE, remove_url = TRUE,
         remove_symbols = TRUE) %>% 
  tokens_wordstem(language = "en") %>% 
  tokens_tolower() %>%
  tokens_select(pattern = stopwords('en'), selection = 'remove') %>%
  tokens_remove(pattern = "(?<=\\d{1,9})\\w+", valuetype = "regex") %>%
  tokens_ngrams(n=2) %>%
  dfm() %>% 
  dfm_trim(min_termfreq = 3) 
topfeatures(Enron_dfm)


topicmodels_Enron_dfm = convert(Enron_dfm, to="topicmodels")

result <- FindTopicsNumber(
  topicmodels_Enron_dfm, 
  topics = seq(from = 5, to = 50, by = 5), 
  method = "Gibbs",
  control = list(seed = 12345),
  verbose = TRUE)

FindTopicsNumber_plot(result)

lda_out = LDA(topicmodels_Enron_dfm, k = 10, method="Gibbs", control = list(seed=12345))

## Find the features with highest values for each topic:
mu_k = tidy(lda_out, matrix = "beta")

top_features = mu_k %>%
  group_by(topic) %>%
  top_n(5,beta) %>% 
  ungroup() %>%
  arrange(topic,-beta)

## Plot the topic and words for easy interpretation
top_features %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()


##############3
#### STM


## First, convert our DFM to a format the works with the STM package:
stm_Enron_dfm = convert(Enron_dfm, to = "stm")

## The STM package has a function to discover the optimal number of topics in a 
## data-driven way (takes a little time):
stm_search_k = searchK(stm_Enron_dfm$documents, stm_Enron_dfm$vocab, 
                       K = seq(10,50,10), prevalence = ~ stm_Enron_dfm$meta$direction)
plot.searchK(stm_search_k)

# prevalnce
out_topics_prevalence = stm(stm_Enron_dfm$documents, stm_Enron_dfm$vocab, 
                            K = 5, prevalence = ~ stm_Enron_dfm$meta$direction, 
                            init.type = "Spectral")
labelTopics(out_topics_prevalence, n=5)
plot(out_topics_prevalence, type = "summary")
