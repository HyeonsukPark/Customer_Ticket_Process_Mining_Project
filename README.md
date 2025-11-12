# Customer_Ticket_Process_Mining_Project 

This README provides an overview of the Customer Ticket Process Mining project, which uses process mining techniques to analyze and optimize the customer support workflow. 

## Project Overview 

This project focuses on data analysis using process mining theory to gain insights into the customer ticket resolution process. The primary goals are to:  

* Discover the actual ticket resolution processes.
* Identify bottlenecks and inefficient paths based on customer satisfaction.
* Visualize the impact of different activities, issue types, and resolvers on customer satisfaction.
* Integrate Large Language Models (LLMs) for automated analysis summaries.

## Key Components  

The project consists of three main files:  

* process_analysis.ipynb (Jupyter Notebook): It contains the core data analysis and process mining discovery. (NOTE: This notebook was created and executed within the Google Colaboratory environment).
   * Process Discovery : It generates process models using standard notations, including Business Process Modeling Notation (BPMN), Process Trees, and Petri Nets.
   * Bottleneck Analysis : It identifies critical issue types and process paths that correlate with low customer satisfaction.
   * Visualization : It uses graphical methods to show which activities, issue types, and resolvers as associated with the highest and lowest satisfaction scores.
   * LLM integration : It uses an LLM (OpenAI) to generate detailed summaries.

* log_df_data: The converted and preprocessed event log data.
* app.py : The main file for the interactive Streamlit dashboard

## Streamlit Dashboard 

The Streamlit application provides an interactive dashboard for visualizing the analysis results. It includes the following components: 
* Bar Graphs : Visualizations based on correlation plots, satisfaction by activity, issue type analysis, and resolver performance.
* Analysis Summary : A section intended for the LLM-generated summary and analysis.

## ⚠️ LLM Integration Note (Action Required)

The deployed Streamlit dashboard currently does not display the LLM-generated summaries due to an issue with the OpenAI API Key being stored as a secret. To enable the full functionality of the dashboard, you must:
1. Generate a valid OpenAI API Key.
2. Add your API Key to the appropriate location in the project code (e.g., within the Streamlit secrets management or the relevant configuration file).






