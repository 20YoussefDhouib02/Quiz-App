// Original array of questions
let allquestions = [
  {
    question: "What does AI stand for?",
    answer: "Artificial Intelligence",
    options: [
      "Artificial Interaction",
      "Advanced Integration",
      "Artificial Intelligence",
      "Automated Intelligence"
    ]
  },
  {
    question: "Which of the following options focuses on identifying and understanding objects?",
    answer: "Computer Vision",
    options: [
      "Expert Systems",
      "Computer Vision",
      "Machine Learning",
      "Natural Language Processing"
    ]
  },
  {
    question: "Which programming language is widely used for AI development due to its simplicity and extensive libraries?",
    answer: "Python",
    options: [
      "JavaScript",
      "Python",
      "Java",
      "C++"
    ]
  },
  {
    question: "Which AI subfield involves teaching machines to learn from past experiences?",
    answer: "Machine Learning",
    options: [
      "Expert Systems",
      "Neural Networks",
      "Robotics",
      "Machine Learning"
    ]
  },
  {
    question: "Which of the following is an example of a personal AI assistant?",
    answer: "Siri",
    options: [
      "Siri",
      "Windows",
      "Wikipedia",
      "Internet Explorer"
    ]
  },
  {
    question: "What does NLP stand for in the AI field?",
    answer: "Natural Language Processing",
    options: [
      "Natural Language Processing",
      "Neuro-Language Parsing",
      "Neural Linguistic Programming",
      "Natural Learning Process"
    ]
  },
  {
    question: "What is overfitting in machine learning?",
    answer: "When a model performs well on training data but poorly on new data",
    options: [
      "When a model performs well on training data but poorly on new data",
      "When a model is unable to find patterns in data",
      "When a model's parameters are under-tuned",
      "When a model has too few features"
    ]
  },
  {
    question: "Which algorithm is used for unsupervised learning tasks?",
    answer: "K-Means Clustering",
    options: [
      "Decision Tree",
      "Linear Regression",
      "K-Means Clustering",
      "Logistic Regression"
    ]
  },
  {
    question: "What does Deep in Deep Learning refer to?",
    answer: "The multiple layers in neural networks",
    options: [
      "The multiple layers in neural networks",
      "The depth of knowledge needed",
      "The amount of data required",
      "The complexity of the problem being solved"
    ]
  },
  {
    question: "Which of the following is an activation function used in neural networks?",
    answer: "ReLU",
    options: [
      "Backpropagation",
      "ReLU",
      "K-Nearest Neighbors",
      "Gradient Descent"
    ]
  },
  {
    question: "What does a confusion matrix help evaluate in a classification task?",
    answer: "The performance of a classification model",
    options: [
      "The distribution of the training data",
      "The learning rate of the model",
      "The performance of a classification model",
      "The speed of the algorithm"
    ]
  },
  {
    question: "Which optimization technique adjusts model weights to minimize the error in machine learning models?",
    answer: "Gradient Descent",
    options: [
      "Naive Bayes",
      "Support Vector Machine",
      "Gradient Descent",
      "Random Forest"
    ]
  },
  {
    question: "In which type of AI are the agents capable of human-like reasoning and can solve problems independently?",
    answer: "General AI",
    options: [
      "General AI",
      "Narrow AI",
      "Supervised AI",
      "Weak AI"
    ]
  },
  {
    question: "Which technique in AI involves training a model on multiple tasks at once, improving the performance on all tasks?",
    answer: "Multitask Learning",
    options: [
      "Transfer Learning",
      "Federated Learning",
      "Meta-Learning",
      "Multitask Learning"
    ]
  },
  {
    question: "Which term refers to AI systems that can generate human-like text, such as GPT?",
    answer: "Generative Models",
    options: [
      "Heuristic Search",
      "Generative Models",
      "Symbolic AI",
      "Discriminative Models"
    ]
  },
  {
    question: "Which machine learning model is often referred to as a universal approximator due to its ability to approximate any continuous function?",
    answer: "Neural Networks",
    options: [
      "Decision Trees",
      "Neural Networks",
      "Random Forest",
      "Support Vector Machines"
    ]
  },
  {
    question: "What is the main goal of transfer learning in AI?",
    answer: "Applying knowledge gained in one task to a different but related task",
    options: [
      "Applying knowledge gained in one task to a different but related task",
      "Collecting more data for training",
      "Improving model accuracy",
      "Reducing model complexity"
    ]
  },
  {
    question: "Which AI model is used for generating images, text, and audio by learning from vast amounts of data?",
    answer: "Generative Adversarial Networks",
    options: [
      "Generative Adversarial Networks",
      "Decision Trees",
      "Recurrent Neural Networks",
      "Convolutional Neural Networks"
    ]
  },
  {
    question: "What does the term 'Big Data' refer to in AI?",
    answer: "Extremely large datasets that are difficult to process using traditional methods",
    options: [
      "Data that comes from sensors",
      "High-speed computations",
      "Large images in machine learning",
      "Extremely large datasets that are difficult to process using traditional methods"
    ]
  },
  {
    question: "What is the purpose of a cost function in machine learning?",
    answer: "To measure how far off a model's predictions are from the actual outcomes",
    options: [
      "To measure how far off a model's predictions are from the actual outcomes",
      "To optimize the dataset",
      "To determine feature importance",
      "To visualize the data"
    ]
  },
  {
    question: "What is the key difference between supervised and unsupervised learning?",
    answer: "Supervised learning uses labeled data while unsupervised learning works with unlabeled data",
    options: [
      "Unsupervised learning requires more memory",
      "Supervised learning uses labeled data while unsupervised learning works with unlabeled data",
      "Unsupervised learning is faster",
      "Supervised learning uses large datasets"
    ]
  },
  {
    question: "Which of these is a challenge in Natural Language Processing?",
    answer: "Understanding context and ambiguity in human language",
    options: [
      "Solving algebraic equations",
      "Processing large datasets",
      "Understanding context and ambiguity in human language",
      "Training models"
    ]
  },
  {
    question: "Which AI subfield is responsible for making machines understand and generate human language?",
    answer: "Natural Language Processing",
    options: [
      "Natural Language Processing",
      "Computer Vision",
      "Neural Networks",
      "Reinforcement Learning"
    ]
  },
  {
    question: "In which type of AI learning do agents learn by receiving rewards or penalties?",
    answer: "Reinforcement Learning",
    options: [
      "Reinforcement Learning",
      "Supervised Learning",
      "Unsupervised Learning",
      "Transfer Learning"
    ]
  },
  {
    question: "What is a common method used for feature scaling in machine learning?",
    answer: "Min-Max Normalization",
    options: [
      "Clustering",
      "Min-Max Normalization",
      "Ensemble Methods",
      "Classification"
    ]
  },
  {
    question: "Which AI technique helps balance accuracy and error by averaging predictions from multiple models?",
    answer: "Ensemble Learning",
    options: [
      "Ensemble Learning",
      "Principal Component Analysis",
      "Reinforcement Learning",
      "Transfer Learning"
    ]
  },
  {
    question: "What is the main objective of hyperparameter tuning in AI?",
    answer: "Optimizing model performance",
    options: [
      "Improving feature selection",
      "Decreasing the dataset size",
      "Reducing training time",
      "Optimizing model performance"
    ]
  },
  {
    question: "Which of these is a popular framework for deep learning?",
    answer: "TensorFlow",
    options: [
      "React",
      "TensorFlow",
      "Angular",
      "Vue.js"
    ]
  },
  {
    question: "Which company developed the AI framework called PyTorch?",
    answer: "Facebook",
    options: [
      "Google",
      "Amazon",
      "Microsoft",
      "Facebook"
    ]
  },
  {
    question: "Which machine learning model uses a tree-like graph of decisions to predict outcomes?",
    answer: "Decision Trees",
    options: [
      "Neural Networks",
      "Random Forests",
      "Decision Trees",
      "Convolutional Neural Networks"
    ]
  }
];

// Function to shuffle the array
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// Shuffle the questions array
for (let nb=0 ; nb<5 ; nb++) {
  shuffleArray(allquestions);
}


// Select the first 5 questions from the shuffled array
let p=Math.random(0,25)
let questions = allquestions.slice(p, p+5);
