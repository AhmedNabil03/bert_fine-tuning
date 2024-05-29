<h1>Fine-Tuning BERT Model on SMS Spam Dataset Using Torch</h1>

<h2>Project Overview</h2>
<p>This project involves fine-tuning a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model to classify SMS messages as either spam or ham (non-spam). The project utilizes the <strong>torch</strong> library for model training and <strong>datasets</strong> from Hugging Face for data handling and processing.</p>

<h2>Dataset</h2>
<p>The dataset used for this project is a collection of SMS messages, labeled as spam or ham. The dataset is read from a CSV file (<strong>spam.csv</strong>) and includes the following preprocessing steps:</p>
<ul>
    <li>Mapping the labels to numerical values (0 for ham and 1 for spam).</li>
    <li>Removing duplicate entries.</li>
    <li>Balancing the dataset by sampling an equal number of ham and spam messages.</li>
    <li>Splitting the dataset into training and testing sets.</li>
</ul>

<h2>Model Training and Evaluation</h2>
<p>The project uses a pre-trained BERT model for sequence classification. The model is fine-tuned with the following steps:</p>
<ol>
    <li><strong>Tokenization:</strong> The text data is tokenized using BERT's tokenizer, ensuring that the input format matches the model's requirements.</li>
    <li><strong>Data Loading:</strong> The tokenized data is loaded into PyTorch <em>DataLoader</em> objects for batch processing.</li>
    <li><strong>Model Configuration:</strong> The BERT model is configured to fine-tune the last two layers while freezing the others.</li>
    <li><strong>Training Loop:</strong> The model is trained over multiple epochs with the AdamW optimizer and CrossEntropyLoss.</li>
    <li><strong>Evaluation:</strong> The trained model is evaluated on the test set using accuracy, precision, recall, F1 score, ROC curve, and confusion matrix.</li>
</ol>

<h2>Results</h2>
<p>The performance of the model is evaluated on the test set, and the results include:</p>
<ul>
    <li><strong>Accuracy:</strong> Measures the overall correctness of the model.</li>
    <li><strong>Precision:</strong> Indicates the proportion of positive identifications that were actually correct.</li>
    <li><strong>Recall:</strong> Indicates the proportion of actual positives that were correctly identified.</li>
    <li><strong>F1 Score:</strong> Harmonic mean of precision and recall.</li>
    <li><strong>ROC Curve and AUC:</strong> Graphical representation of the model's performance across different thresholds.</li>
    <li><strong>Confusion Matrix:</strong> Detailed breakdown of true positives, true negatives, false positives, and false negatives.</li>
</ul>

<h2>Key Components</h2>
<ul>
    <li><strong>Data Preprocessing:</strong> Cleaning, tokenization, and formatting of the SMS dataset.</li>
    <li><strong>Model Fine-Tuning:</strong> Adjusting the pre-trained BERT model for SMS spam classification.</li>
    <li><strong>Evaluation Metrics:</strong> Calculation and visualization of accuracy, precision, recall, F1 score, ROC curve, and confusion matrix.</li>
</ul>

<h2>Conclusion</h2>
<p>This project demonstrates the application of a BERT model for text classification tasks, specifically in detecting SMS spam. By fine-tuning the model on a balanced dataset, the project achieves robust performance metrics, illustrating the effectiveness of transfer learning with transformer models.</p>
