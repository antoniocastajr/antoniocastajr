## 🧑‍💻 Antonio's coding path 🛣️

My journey into the world of coding began during my dual bachelor’s degree in Computer Science and Business, driven by a passion to understand every corner of programming. I started with Java, where I learned the foundations of data structures, classes, objects, and inheritance. Building on this base, I explored Assembler and C, guided by courses such as Operating Systems and Distributed Systems, which helped me see how these languages connect to the inner workings of computing. In the later years of my degree, I turned my focus to Python and the field of Data Science, an area that perfectly bridges my technical and business background, while also fueling my curiosity. Today, I continue this path by sharing projects on Data Science, AI Agents, and Large Language Models (LLMs) here on GitHub, documenting everything I learn along the way

---
### 🧰 Languages, Libraries and Tools

<img align="left" alt="Python" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-plain.svg" />
<img align="left" alt="Java" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/java/java-original.svg"/>
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/apachespark/apachespark-original-wordmark.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/npm/simple-icons@v15/icons/langgraph.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original-wordmark.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tensorflow/tensorflow-original.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytorch/pytorch-plain-wordmark.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/npm/simple-icons@v15/icons/ollama.svg" />
<img align="left" alt="Git" width="30px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/npm/simple-icons@v15/icons/openai.svg" />

<br />

---

## 🧑‍💻 Antonio's Projects 🔬

#### 🔭 Currently Working On  

- **[Multi-Agent Marketing Platform for Business Intelligence, SQL Automation & Email Targeting][marketing]** → Available at https://marketing-analyst.streamlit.app/.  This project demonstrates the **capabilities of Machine Learning and Artificial Intelligence in marketing analytics.** By using language models, we can generate SQL queries, analyze customer segments, and create targeted marketing strategies.

  In detail, this multi-agent AI system implements the following specialized agents:
  - 📋 **Data Overview:** Provides an **introduction to the database** used in this project. Displays column names, data types, brief descriptions of each table, and visualizations to complement the analysis.
  - 🔍 **Data Exploration:** It a **SQL generator that translates natural language into database queries** and interprets insights automatically.
  - 💼 **Business Analysis:** Produces a **business report based on key metrics** such as revenue, average order value, and customer lifetime value (CLV).
  - 📧 **Email Marketing:** Identifies target customers and **creates personalized email campaigns.**
  - 📣 **Marketing Strategy:** Segments customers into clusters and **generates targeted marketing strategies for each group.**

The following demo shows the performance of the following agents; Data Exploration, Email Marketing, and Marketing Strategy.

<p align="center">
  <video
    src="https://github.com/user-attachments/assets/60fcc44f-3ac5-4367-8383-5b436f806943"
    controls
    muted
    playsinline
    loop
    width="720" 
    height="405">
  </video>
</p>

- 🤖 **[Naviria: A Telegram AI Agent Powered by LangGraph, RAG, and MCP][naviria]** → Naviria is a conversational AI agent deployed on Telgram I’m building to learn and showcase the core technologies behind modern AI agents. It combines LLM reasoning by [LangGraph][langgraph], [Model Context Protocol (MCP)][mcp] for pluggable tools, and [Retrieval-Augmented Generation (RAG)][rag] with embeddings and a vector store. In detail, Naviria integrates the next selection of AI Agent's resources:

  - 🧩 **Advanced agentic logic:** Built on [LangGraph][langgraph], Naviria runs as a state machine, intelligently routing user requests across tools and knowledge sources.
  - 🔎 **Retrieval-Augmented Generation (RAG):** A self-updating knowledge base powered by a [FAISS][faiss] vector store and Google’s high-performance [EmbeddingGemma][embedder] model for document embeddings.
  - 🛰️ **Model Context Protocol (MCP):** Seamlessly connects to external tools like [Tavily Search][tavily] to fetch fresh web data and enrich the knowledge base in real time.
  - 🧠 **Personalized long-term memory:** Maintains persistent [memory][memory] per user for contextual, personalized conversations.
  - 🔀 **Flexible LLM integration:** Works with multiple models—OpenAI (e.g., [gpt-5-nano][gpt5nano], [gpt-4.1-nano][gpt41nano], [gpt-4o-mini][gpt4omini]) and open-source via Ollama (e.g., [llama3.1:8b][llama], [gpt-oss:20b][gtposs]).

<img width="3840" height="302" alt="Untitled diagram _ Mermaid Chart-2025-09-08-184056" src="https://github.com/user-attachments/assets/2f97db86-8f36-4b80-8d26-7b21bfb6fa64" />

<p align="center">
  <video
    src="https://github.com/user-attachments/assets/03abb5a3-eb72-4c82-a20d-348b1484bc75"
    controls
    muted
    playsinline
    loop
    width="720">
  </video>
</p>

#
#### ✋ On Hold

- 📘 **[Machine Learning Algorithms from Scratch][machine]** → A hands-on project where I build classic **machine learning algorithms completely from scratch,** while also **breaking down the math and theory behind them.** So far, I’ve implemented Linear Regression, Logistic Regression, Naive Bayes, and PCA and **compare my results with Sklearn.** This is an ongoing project, and whenever I get some free time, I plan to expand it with Random Forest, SVM, and KNN.

#
#### 🌟 Projects that you should take a peak at

- 🧾 **[Fraud Detection Using PySpark][fraud]** → Credit card transactions were classified into fraud and non-fraud by analyzing client behavior through a **variety of machine learning models,** including Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM. To support this process, **features were engineered and selected using correlation analysis, visual inspection, and chi-square tests,** ensuring that only the most relevant attributes were retained. These inputs were then integrated into end-to-end **pipelines,** which streamlined data transformations and automated predictions.

  Building on these foundations, the **three best performing models (Random Forest, XGBoost, and LightGBM) were combined into an ensemble using majority voting. To provide transparency and deeper insights, model interpretability was finally addressed using the **SHAP library,** which explained feature contributions and the reasoning behind predictions.
  
  The results are shown as follows:

| Metric          | Naive Bayes | Logistic Regression | Random Forest | SVC    | XGBoost | LightGBM | Majority Vote (RF+XGB+LGBM) |
|-----------------|:-----------:|:-------------------:|:-------------:|:------:|:-------:|:--------:|:---------------------------:|
| **Accuracy**    |   0.6565    |        0.9906       |   **0.9980**  | 0.9873 |  0.9934 |  0.9965  |           0.9976            |
| **Precision**   |   0.0094    |        0.2297       |   **0.8063**  | 0.1837 |  0.3632 |  0.5280  |           0.6402            |
| **Recall**      |   0.8410    |        0.6061       |     0.6462    | 0.6629 |**0.9552**|  0.8737  |           0.8741            |
| **Specificity** |   0.6558    |        0.9921       |   **0.9994**  | 0.9886 |  0.9935 |  0.9970  |           0.9981            |
| **F1-Score**    |   0.0186    |        0.3332       |     0.7174    | 0.2877 |  0.5263 |  0.6582  |         **0.7391**          |

<p align="center">
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/f9518ac7-3f87-4e39-bee5-cf2589e222ad" />
</p>

- 📝 **[Pre-training and Fine-tuning BERT from Scratch for Movie Review Classification][bert]** → BERT was the first deeply bidirectional model, unlike previous models that processed text only from left-to-right or right-to-left. To achieve this bidirectionality, it introduced two novel pre-training objectives: **Masked Language Modeling (MLM)**, where random tokens are hidden and predicted to capture context from both sides, and **Next Sentence Prediction (NSP),** which trains the model to understand sentence relationships.

  Pre-training, including MLM and NSP, was performed from scratch using the **[BookCorpus dataset.][bookcorpus]** Futhermore, BERT was then fine-tuned on the **[IMDB dataset][imdb]** (50,000 balanced movie reviews) for sentiment classification by adding a classification head on top of the transformer layers. The model achieved an **F1-Score of 0.7947.**

  <img width="3840" height="585" alt="image" src="https://github.com/user-attachments/assets/aa4f3901-0174-4996-b4d5-d59af5e1b87e" />

  <p align="center">
  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/9969913f-4839-424a-bbf6-4c0c3db2c9d5" />
  </p>

  
- 🩺 **[Melanoma Detection with Convolutional Neural Networks (CNN)][melanoma]** → Project developed for CS 512: Computer Vision at IIT. This repository implements **two different CNN architectures in Keras to detect melanoma cases.** The first model is inspired by the work of Aya Abu Ali and Hasan Al-Marzouqi presented at the [ICECTA Conference][icecta]. The second model was designed and implemented by me, **achieving a 11.38% improvement in F1-Score compared to the baseline, reaching 0.87.** To enhance performance, techniques such as **data augmentation and sampling were applied during training.**

  <img width="1382" height="888" alt="image" src="https://github.com/user-attachments/assets/df281a17-2a3d-4878-958e-4b834e8d197d" />

- 💬 **[Sentiment Analysis with Machine Learning Algorithms][sentiment]** →


[naviria]: https://github.com/antoniocastajr/Naviria
[marketing]: https://github.com/antoniocastajr/Marketing-Analyst
[mcp]: https://modelcontextprotocol.io/docs/getting-started/intro
[rag]: https://arxiv.org/pdf/2005.11401
[langgraph]: https://python.langchain.com/docs/introduction/
[faiss]: https://python.langchain.com/docs/integrations/vectorstores/faiss/
[embedder]: https://huggingface.co/blog/embeddinggemma
[tavily]: https://docs.tavily.com/documentation/mcp
[memory]: https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/
[gpt5nano]: https://platform.openai.com/docs/models/gpt-5-nano
[gpt41nano]: https://platform.openai.com/docs/models/gpt-4.1-nano
[gpt4omini]: https://platform.openai.com/docs/models/gpt-4o-mini
[llama]: https://ollama.com/library/llama3.1
[gtposs]: https://ollama.com/library/gpt-oss
[fraud]: https://github.com/antoniocastajr/Fraud-Detection-Using-PySpark
[machine]: https://github.com/antoniocastajr/Machine-Learning-Algorithms-from-Scratch
[bert]: https://github.com/antoniocastajr/BERT-from-Scratch
[bookcorpus]: https://huggingface.co/datasets/rojagtap/bookcorpus
[imdb]: https://www.kaggle.com/datasets/mahmoudshaheen1134/imdp-data
[melanoma]: https://github.com/antoniocastajr/Computer-Vision/tree/main/Project
[icecta]: https://github.com/antoniocastajr/Computer-Vision/blob/main/Project/project/sources/Melanoma%20detection.pdf
[sentiment]: https://github.com/antoniocastajr/Machine-Learning/tree/main/Project
[gini]: https://github.com/antoniocastajr/SaturdaysAI
