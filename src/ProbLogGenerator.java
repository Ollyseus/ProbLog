 import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import libsvm.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;




public class ProbLogGenerator {

	static BufferedReader reader = null;
	static Instances testData;
	static Instances trainingData;
	
	static Instances labeled;
	
	static ArrayList<Double> dates;
	
	static NaiveBayes nb;
	static J48 j;
	static IBk knn;
	
	static Classifier classifier;
	
	
	public static void main(String[] args) {
		
		readTestData();
		readTrainingData();
		constructClassifier();
		evaluate();
		classify();
		
		createProbLog();
	}
	
	public static void readTestData(){
		
		
		try {
			reader = new BufferedReader(new FileReader("C:/LunaWorkspace/ProbLog/assets/Robert/test.arff"));	
			testData = new Instances(reader);
			reader.close();
			
			// setting class attribute
			testData.setClassIndex(testData.numAttributes()-1);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void readTrainingData(){
		
		
		try {
			reader = new BufferedReader(new FileReader("C:/LunaWorkspace/ProbLog/assets/Oliver/train.arff"));	
			trainingData = new Instances(reader);
			reader.close();
			
			// setting class attribute
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void constructClassifier() {
		
		nb = new NaiveBayes();
		j = new J48();
		knn = new IBk();
		Id3 id3 = new Id3();
		
		RandomForest r = new RandomForest();
		
		classifier = r;
				
		dates =  new ArrayList<Double>();
		
		//save date
		for(int i=0;i<testData.numInstances();i++){
			dates.add(testData.instance(i).value(testData.numAttributes()-2));
		}
		
		
		// remove timestamp from test data, using filter
	    try {
			 String[] options = new String[2];
			 options[0] = "-R";                                    // "range"
			 options[1] = testData.numAttributes() -1+"";			//first attribute
			 Remove remove = new Remove();                         // new instance of filter
			 remove.setOptions(options);                           // set options
			 remove.setInputFormat(testData);                          // inform filter about dataset **AFTER** setting options
			 testData = Filter.useFilter(testData, remove); // apply filter
		} catch (Exception e1) {
			e1.printStackTrace();
		}  
	    
		// remove timestamp from training data, using filter
	    try {
			 String[] options = new String[2];
			 options[0] = "-R";                                    // "range"
			 options[1] = trainingData.numAttributes() - 1+"";			//remove timestamp
			 Remove remove = new Remove();                         // new instance of filter
			 remove.setOptions(options);                           // set options
			 remove.setInputFormat(trainingData);                          // inform filter about dataset **AFTER** setting options
			 trainingData = Filter.useFilter(trainingData, remove); // apply filter
		} catch (Exception e1) {
			e1.printStackTrace();
		} 
		
	    System.out.println(trainingData.toString());
	    
		
		try {
			classifier.buildClassifier(trainingData);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println(j.toString());
		System.out.println(classifier.toString());
	}

	public static void evaluate() {
			
		 try {
		    // evaluate classifier and print some statistics
			Evaluation eval = new Evaluation(trainingData);
			eval.evaluateModel(classifier, testData);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	public static void classify() {
		
		 // create copy
		 labeled = new Instances(testData);
		 
		 // label instances
		 for (int i = 0; i < labeled.numInstances(); i++) {
		 
			 
		 //create Distribution
				   double clsLabel = 0;
				try {
					clsLabel = classifier.classifyInstance(labeled.instance(i));
					labeled.instance(i).setClassValue(clsLabel);
					double[] probs = null;
					probs = classifier.distributionForInstance( labeled.instance(i));
				} catch (Exception e) {
					e.printStackTrace();
				}
	
		 }
		 
		 // save labeled data
		 BufferedWriter writer = null;
		try {
			writer = new BufferedWriter( new FileWriter("C:/LunaWorkspace/ProbLog/assets/Oliver/test_labeled.arff"));
			writer.write(labeled.toString());
			writer.newLine();
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void createProbLog() {
		
		BufferedWriter writer = null;
		try {
			writer = new BufferedWriter( new FileWriter("C:/LunaWorkspace/ProbLog/assets/problog.csv"));
			String header= "timestamp"+","+"p(sitting)"+","+"p(walking)"+","+"p(standing)"+","+"p(lying)"+","+"p(on table)"+","+"p(running)" + ","+ "predicted,actual";
			writer.write(header);
			writer.newLine();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		 for (int i = 0; i < labeled.numInstances(); i++) {
			 
			 		//get probabilities for current labeled instance
				   double[] probs = null;
				   
					try {
						probs = classifier.distributionForInstance( labeled.instance(i));
					} catch (Exception e) {
						e.printStackTrace();
					}
					
 
					try {
							//get proper unix timestamp
							//String timestamp = labeled.instance(i).value(12)+"";
							String timestamp = dates.get(i)+"";
							
							//get actual class from testset
							String predicted = labeled.instance(i).value(labeled.numAttributes()-1)+"";
							String actual = testData.instance(i).value(testData.numAttributes()-1)+"";
							
							
						//	timestamp = timestamp.replace(".", "");
						//	timestamp = timestamp.replace("E12", "");
							
							//write time and probabilities to csv file
							String s = timestamp +","+probs[0]+","+probs[1]+","+probs[2]+","+probs[3]+","+probs[4]+","+probs[5]+","+predicted+","+actual;
							System.out.println(s);
							writer.write(s);
							writer.newLine();
					} catch (IOException e) {
							e.printStackTrace();
					}
					
//					System.out.println(labeled.instance(i).value(12));
//					
//					for(int index=0;index<probs.length;index++){
//						System.out.print(Array.getDouble(probs, index) + " , ");
//					}
//					System.out.println();
		 }
		 
			try {
				writer.flush();
				writer.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
	}
	
}
