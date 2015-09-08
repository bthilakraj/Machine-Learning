package com.ml.decisionTree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import org.codehaus.jackson.map.ObjectMapper;
import org.codehaus.jackson.map.ObjectWriter;


/**
 * @author Thilak
 * @Main Class : Start of the Program which holds all the calls-->Constructing the tree and 
 * Parsing the training and validation data into the tree and calculating the accuracies with confidence level 0%,95%,99%
 */
public class InputMain {
	static String trainingData = null;
	static String validationData = null;
	static Properties prop = new Properties();
    static InputStream input = null;
	public static void main(String[] args) {
		
	
		try {
			prop = new Properties();
		    prop.load(Thread.currentThread().getContextClassLoader().getResourceAsStream("Constants.properties"));
		    double csv95=Double.parseDouble(prop.getProperty("CSV95"));	
		    double csv0=Double.parseDouble(prop.getProperty("CSV0"));	
		    double csv99=Double.parseDouble(prop.getProperty("CSV99"));	
		   //  trainingData="training.txt";
		   // validationData="validation.txt";
			trainingData = "C:/Users/Thilak/Desktop/ml/hw1/training.txt";
			validationData = "C:/Users/Thilak/Desktop/ml/hw1/validation.txt";
			ArrayList<Data> trainingDataList = getInputDataList(trainingData);
			ArrayList<Data> validationDataList = getInputDataList(validationData);
			Node nodea=new Node();
			nodea.setAttributeNumber(-1);
			nodea.setAttributeValue("parentRoot");
			Dtree tree=new Dtree();
			System.out.println("----------CHI SQUARE VALUE : 95 %-----------------");
			Node node1=new Node();
			node1=tree.constructTreeUsingInformatonGain(trainingDataList,nodea,csv95);
			Node nodeb=new Node();
			nodeb.setAttributeNumber(-1);
			nodeb.setAttributeValue("parentRoot");
			Dtree tree1=new Dtree();
			Node node2=new Node();
			node2=tree1.constructTreeUsingMisclassification(trainingDataList,nodeb,csv95);
			ObjectWriter ow = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json = ow.writeValueAsString(node1);
			System.out.println("TREE USING ID3 entropy");
			System.out.println(json);
			accuracyCheck(trainingDataList,node1);
			accuracyCheck1(validationDataList,node1);
			ObjectWriter ow1 = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json1 = ow1.writeValueAsString(node2);
			System.out.println("---------------------------");
			System.out.println("\nTREE USING MISCLASSIFICATION IMPURITY");
			System.out.println(json1);
			accuracyCheck(trainingDataList,node2);
			accuracyCheck1(validationDataList,node2);
			System.out.println("----------CHI SQUARE VALUE : 0 %-----------------");
			Node nodec=new Node();
			nodec.setAttributeNumber(-1);
			nodec.setAttributeValue("parentRoot");
			Dtree tree2=new Dtree();
			Node node3=new Node();
			node3=tree2.constructTreeUsingInformatonGain(trainingDataList,nodec,csv0);
			Node noded=new Node();
			noded.setAttributeNumber(-1);
			noded.setAttributeValue("parentRoot");
			Dtree tree3=new Dtree();
			Node node4=new Node();
		    node4=tree3.constructTreeUsingMisclassification(trainingDataList,noded,csv0);
			ObjectWriter ow2 = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json2 = ow2.writeValueAsString(node3);
			System.out.println("TREE USING ID3");
			System.out.println(json2);
			accuracyCheck(trainingDataList,node3);
			accuracyCheck1(validationDataList,node3);
			ObjectWriter ow3 = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json3 = ow3.writeValueAsString(node4);
			System.out.println("---------------------------");
			System.out.println("\nTREE USING MISCLASSIFICATION IMPURITY");
			System.out.println(json3);
			accuracyCheck(trainingDataList,node4);
			accuracyCheck1(validationDataList,node4);
			System.out.println("----------CHI SQUARE VALUE : 99 %-----------------");
			Node nodee=new Node();
			nodee.setAttributeNumber(-1);
			nodee.setAttributeValue("parentRoot");
			Dtree tree4=new Dtree();
			Node node5=new Node();
			node5=tree4.constructTreeUsingInformatonGain(trainingDataList,nodee,csv99);
			Node nodef=new Node();
			nodef.setAttributeNumber(-1);
			nodef.setAttributeValue("parentRoot");
			Dtree tree5=new Dtree();
			Node node6=new Node();
			node6=tree5.constructTreeUsingMisclassification(trainingDataList,nodef,csv99);
			ObjectWriter ow4 = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json4 = ow4.writeValueAsString(node5);
			System.out.println("TREE USING ID3");
			System.out.println(json4);
			accuracyCheck(trainingDataList,node5);
			accuracyCheck1(validationDataList,node5);
			ObjectWriter ow5 = new ObjectMapper().writer().withDefaultPrettyPrinter();
			String json5 = ow5.writeValueAsString(node6);
			System.out.println("---------------------------");
			System.out.println("\nTREE USING MISCLASSIFICATION IMPURITY");
			System.out.println(json5);
			accuracyCheck(trainingDataList,node6);
			accuracyCheck1(validationDataList,node6);
			
		} catch (Exception e) {
			e.printStackTrace();
		}	
	}

	



	/**ACCURACYCHECK1:Method used for calculating the number of classified and misclassified data of Validation dataset
	 * @param validationDataList
	 * @param node1
	 */
	private static void accuracyCheck1(ArrayList<Data> validationDataList,
			Node node1) {
		int classified=0;
		int misclassified=0;
		for (Data data : validationDataList) {
			if(treeTraverse(data,node1)){
				classified=classified+1;
			}else{
				misclassified=misclassified+1;
			}
		}
		double result = classified+misclassified;
		double percentage=((classified/result)*100);
		System.out.println("Classification of VALIDATION DataSet");
		System.out.println("classified:"+classified+" : misclassified:"+misclassified);
		System.out.println("ACCURACY PERCENTAGE:"+percentage);
		
	}





	/**getInputDataList: Method used to read the text file of training and testing dataset and
	 * storing it in an Arraylist of <Data>format which hold the classtype and the data[].each 
	 * character is stored in the array[56] 
	 * @param inputData
	 * @return
	 */
	private static ArrayList<Data> getInputDataList(String inputData) {
		// TODO Auto-generated method stub
		ArrayList<Data> data = new ArrayList<Data>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(inputData));	
			String line=null;
			while ((line = br.readLine())!=null) {
				Data data1=null;
				String[] dataParts = line.split(" ");
				String classTypeValue = dataParts[1];
				String dnaSequence = dataParts[0];
				char[] splitInputData=null;
				if (classTypeValue.contains("+")){
					data1 = new Data();
					data1.setClassType("Promoter");
					if (dnaSequence.length() != Integer.parseInt(prop.getProperty("NoOfAttributes"))){
						System.out.println("Input Data Error");
					}
					else{
				    splitInputData = new char[57];
					for (int i = 0; i < dnaSequence.length(); i++) {
					splitInputData[i]=dnaSequence.charAt(i);
					}
					data1.setAttributeValue(splitInputData);	
				}
				}else{
					data1 = new Data();
					data1.setClassType("Non-Promoter");
					if (dnaSequence.length() != Integer.parseInt(prop.getProperty("NoOfAttributes"))){
						System.out.println("Input Data Error");
					}
					else{
						splitInputData = new char[57];
						for (int i = 0; i < dnaSequence.length(); i++) {
						splitInputData[i]=dnaSequence.charAt(i);
						}
					data1.setAttributeValue(splitInputData);
				}
				}
				data.add(data1);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		return data;
		
	}
	
	
	/**accuracyCheck:Method used for calculating the number of classified and misclassified data of Training dataset
	 * @param trainingDataList
	 * @param node1
	 */
	private static void accuracyCheck(ArrayList<Data> trainingDataList,
			Node node1) {
		// TODO Auto-generated method stub
		int classified=0;
		int misclassified=0;
		for (Data data : trainingDataList) {
			if(treeTraverse(data,node1)){
				classified=classified+1;
			}else{
				misclassified=misclassified+1;
			}
		}
		double result = classified+misclassified;
		double percentage=((classified/result)*100);
		System.out.println("Classification of TRAINING DataSet");
		System.out.println("classified:"+classified+" : misclassified:"+misclassified);
		System.out.println("ACCURACY PERCENTAGE:"+percentage);
	}
	/**treeTraverse: Method used to traverse the constructed tree to go towards the leaf node and check the value if it returns true.
	 * @param data
	 * @param node1
	 * @return true or false value as per the traverse
	 */
	private static boolean treeTraverse(Data data, Node node1) {
		// TODO Auto-generated method stub
		Node parentNode = node1;	
		boolean b=true;
		while (b) {
			boolean f = true;	
			for (Node node11 : parentNode.node) {	
				int a=node11.getAttributeNumber();
				if(String.valueOf(data.getAttributeValue(a)).equalsIgnoreCase(node11.getAttributeValue())) { 
					f=false;		
					if (node11.node.size()==0) {	
						if (node11.getClassType().equalsIgnoreCase(data.getClassType())) { 
							return true;		
						}else {
							return false;		
						}
					}else{						
						parentNode = node11;
					}
					break;
				}				
			}
			if (f) {			
				return false;
			}
		}
		return b;
	}
}
