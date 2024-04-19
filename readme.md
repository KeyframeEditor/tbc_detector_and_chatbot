<h1 align="center" id="title">AI-ENABLED computer vision for TUBERCULOSIS DETECTION FROM MRI SCANS</h1>

<p align="center"><img src="https://socialify.git.ci/KeyframeEditor/tbc_detector_and_chatbot/image?font=Inter&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Formal%20Invitation&amp;theme=Dark" alt="project-image"></p>

<p id="description">Streamlit project to scan MRI Scan images for TBC and chat with RAG chatbot around TBC in Indonesia. The project was developed by Andhika Rahmanu Bayu Rahmat Wibowo and Tabitha Andrea.</p>

  
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Detect TBC/Normal lungs based on MRI Scan Image
*   Chatbot to talk around TBC cases in Indonesia (Retrieval-Augmented Generation)
*   No need to setup Support Vector DB or S3
*   Runs on memory. This project resets itself after restart/shutdown

<h2>🛠️ Installation Steps:</h2>

<p>1. Setup AWS CLI for your local machine</p>

<p>2. Manually install all necessary libraries. There is no requirements.txt at the moment. I'm to lazy to do it</p>

<p>3. Running the App</p>

```
streamlit run 1_📷_Scan_MRI_Image.py --server.port 8080
```

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Developed & tested with AWS Cloud9
*   AWS Bedrock
*   Anthropic's Claude-3 Model
*   Streamlit Python Framework
*   LangChain Framework
*   TensorFlow Keras Model