package com.ml.decisionTree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;



/**
 * @author Thilak
 * Dtree Class holds all methods to construct the decision tree .It includes calculation of entropy, 
 * misclassification impurity and Information Gain , deciding root node, get subset of data
 */
public class Dtree {


	/**getDiscreteLists Method: To Get the subset of data in order to further iterate to the tree 
	 *
	 * @param trainingDataList(Input list-->Training and Validation data )
	 * @param i-> Attribute Id
	 * @return Subset Discrete data of format<SubsetDiscreteData> holding Value(a,g,c,t) and Its respective ArrayList<data>
	 */
	private ArrayList<SubsetDiscreteData> getDiscreteLists(
			ArrayList<Data> trainingDataList, int i) {
		// TODO Auto-generated method stub
		ArrayList<SubsetDiscreteData> discreteData= new ArrayList<SubsetDiscreteData>();
		ArrayList<Data> dataA= new ArrayList<Data>();
		ArrayList<Data> dataG= new ArrayList<Data>();
		ArrayList<Data> dataC= new ArrayList<Data>();
		ArrayList<Data> dataT= new ArrayList<Data>();
		for (Data data : trainingDataList) {
			char s = data.getAttributeValue(i);
			if (s=='a') {
				dataA.add(data);
			}else if (s=='g') {
				dataG.add(data);
			}
			else if (s=='c') {
				dataC.add(data);	
			}
			else if (s=='t') {
				dataT.add(data);
			}
			
		}
		SubsetDiscreteData discreteA=new SubsetDiscreteData();
		discreteA.setValue("a");
		discreteA.setData(dataA);
		discreteData.add(discreteA);
		SubsetDiscreteData discreteG=new SubsetDiscreteData();
		discreteG.setValue("g");
		discreteG.setData(dataG);
		discreteData.add(discreteG);
		SubsetDiscreteData discreteC=new SubsetDiscreteData();
		discreteC.setValue("c");
		discreteC.setData(dataC);
		discreteData.add(discreteC);
		SubsetDiscreteData discreteT=new SubsetDiscreteData();
		discreteT.setValue("t");
		discreteT.setData(dataT);
		discreteData.add(discreteT);
		return discreteData;
	}
	
	
	/**constructTreeUsingInformatonGain Method: Constructing the tree with the Entropy Calculation and Information Gain and 
	 * Selecting the attribute which has the highest Information Gain as the root node  and the iterating through its possible values(a,g,c,t)
	 * and calling the method recursively to builds the tree.
	 * @param trainingDataList - iNput List:Training.txt,Validation.txt
	 * @param parenNode
	 * @param csv: Chisquare Value for 0,95,99 
	 * @return
	 */
	public Node constructTreeUsingInformatonGain(
			ArrayList<Data> trainingDataList, Node parenNode, double csv) {
		// TODO Auto-generated method stub
		double entropyValue;
		entropyValue = entropyCalculation(trainingDataList);	
		if (entropyValue == 0) {						
			parenNode.setClassType(trainingDataList.get(0).getClassType());	
			return parenNode;									
		}else{
			int attributeNumber = highInformationgainAttribute(trainingDataList);	
			if (getThreshold(trainingDataList,attributeNumber) > csv) {
				ArrayList<SubsetDiscreteData> subset = getDiscreteLists(trainingDataList, attributeNumber);
				Iterator it1 = subset.iterator();
				while (it1.hasNext()) {	
					SubsetDiscreteData element = (SubsetDiscreteData) it1.next();
					ArrayList<Data> list =element.getData();
					Node childNode = new Node();
					childNode.setAttributeNumber(attributeNumber);
					childNode.setAttributeValue(element.getValue());
					parenNode.addNextNodeData(childNode);			
					constructTreeUsingInformatonGain(list,childNode,csv);		
				}
			}else{	
				parenNode.setClassType(setMaxtoLeaf(trainingDataList));	
				return parenNode;												}
		}
		
		return parenNode;
	}

	/**setMaxtoLeaf Method: If the decision Node ChiSquare threshold value is lesser than the Chi sQaure value with respect
	 * to Confidence Level (0%,95%,99%)
	 * then not further checking and growing the tree.Pruning is done. then we need to have the leaf node.
	 * Leaf node is identified depending upon maximum number of 
	 * promoters or non promoters.
	 * @param trainingDataList
	 * @return if promoters are more then leaf node is promoter and vice-versa
	 */
	private String setMaxtoLeaf(ArrayList<Data> trainingDataList) {
		int promoterCount = 0;
		int nonPromoterCount =0;
		String result=null;
		for (Data data : trainingDataList) {
			if (data.getClassType().equalsIgnoreCase("Promoter")) {	
				promoterCount++;			
			}else{
				nonPromoterCount++;			
			}
		}
		if(promoterCount>nonPromoterCount){
			result="Promoter";
		}
		else{
			result="Non-Promoter";
		}
					
		return result;
		
	}



	/**entropyCalculation Method:Calculate Entropy using the formula(-i/n log2 (i/n))
	 * @param trainingDataList
	 * @return double entropy value
	 */
	private double entropyCalculation(ArrayList<Data> trainingDataList) {
		// TODO Auto-generated method stub

		double entropyValue= 0;
		double entropy1=0;
		double entropy2=0;
		ArrayList<Data> promoterList=new ArrayList();
		ArrayList<Data> nonPromoterList=new ArrayList();
		for(Data data:trainingDataList){
			if(data.getClassType().equalsIgnoreCase("Promoter")){
				promoterList.add(data);
			}
			else{
				nonPromoterList.add(data);
			}
		}
		double dataSize = trainingDataList.size();
		double promoterSize = promoterList.size();
		double nonPromoterSize = nonPromoterList.size();
			if (promoterSize==0 || promoterSize==dataSize) {
				return 0;
			}
			else{
				entropy1 = -((promoterSize/dataSize)*logBase2(promoterSize/dataSize));
			}
			if (nonPromoterSize==0 || nonPromoterSize==dataSize) {
				return 0;
			}
			else{
				entropy2 = -((nonPromoterSize/dataSize)*logBase2(nonPromoterSize/dataSize));
		}
			entropyValue=entropy1+entropy2;
			
		return entropyValue;
	}

	/**logBase2 MEethod: Calculation of Log Base 2 which is used for Entropy Calculation
	 * @param d
	 * @return double logBase2 value
	 */
	private double logBase2(double d) {
		// TODO Auto-generated method stub
		return Math.log(d)/Math.log(2);
	}

	/**getThreshold method:Calculation of Threshold using Chi Square Formaula using the Exact output value and Expected Output Value
	 * and The value generated is compared with the Chi Square Table value for Split Stopping
	 * Split Stopping Formula as per the project question is built using the formula given in the question
	 * @param trainingDataList
	 * @param attId
	 * @return Threshold value
	 */
	private double getThreshold(ArrayList<Data> trainingDataList, int attId) {
		// TODO Auto-generated method stub
		double chiValue = 0;
		ArrayList<SubsetDiscreteData> discreteData= getDiscreteLists(trainingDataList,attId);
		Map<String, List<Data>> map = getMapData(trainingDataList);	
		double p = map.get("Promoter").size();		
		double n = map.get("Non-Promoter").size();				
		Iterator it = discreteData.iterator();
		while (it.hasNext()) {	
			SubsetDiscreteData element = (SubsetDiscreteData) it.next();
			ArrayList<Data> dataSub= (ArrayList<Data>) element.getData();
			Map<String, List<Data>> rm = getMapData(dataSub);	
			double pi = rm.get("Promoter").size();		
			double ni = rm.get("Non-Promoter").size();		
			double p_i = p*((pi+ni)/(p+n));
			double n_i = n*((pi+ni)/(p+n));
			chiValue += (((pi-p_i)*(pi-p_i)/p_i) + ((ni-n_i)*(ni-n_i)/n_i)  ); 
		}
		return chiValue;
	}
	
	
	/**getMapData Method:Returns a map by grouping Promoters and Non promoters to calculate its count
	 * @param trainingDataList
	 * @return map
	 */
	private Map<String, List<Data>> getMapData(ArrayList<Data> trainingDataList) {
		// TODO Auto-generated method stub
		List<Data> promoterCount = new ArrayList<Data>();
		List<Data> nonPromoterCount = new ArrayList<Data>();
		for (Data data : trainingDataList) {
			if (data.getClassType().equalsIgnoreCase("Promoter")) {	
				promoterCount.add(data);			
			}else{
				nonPromoterCount.add(data);			
			}
		}
		Map<String, List<Data>> map = new HashMap<String, List<Data>>();
		map.put("Promoter", promoterCount);					
		map.put("Non-Promoter", nonPromoterCount);			
		return map;
	}
	

	/**highInformationgainAttribute METHOD:Used to Identify the Attribute Id with Highest Information Gain to set it as a Root NODE
	 * @param trainingDataList
	 * @return aTTRIBUTE nUMBER with Highest Information Gain
	 */
	private int highInformationgainAttribute(ArrayList<Data> trainingDataList) {
		// TODO Auto-generated method stub
		double infomationGain=-9999999;				
		int attributeNodePosition=0;	
		boolean a=true;
		char[] sdata = trainingDataList.get(0).getAttributeValue();	
		for(int i=0;i<sdata.length;i++) {						
			if (a) {
				attributeNodePosition=i;
				a = false;
			}
			double infoGain = informationGainCalculation(trainingDataList,i);
			if (infoGain>infomationGain) {							
				infomationGain = infoGain;
				attributeNodePosition = i;					
			}
		}
		return attributeNodePosition;
	}


	/**informationGainCalculation METHOD: Gain(S, A) = Entropy(S) - S ((|Sv| / |S|) * Entropy(Sv)) is calculated using this method
	 * @param trainingDataList
	 * @param i-Attribute Id
	 * @return Information Gain
	 */
	private double informationGainCalculation(ArrayList<Data> trainingDataList,
			int i) {
		// TODO Auto-generated method stub
		double n = trainingDataList.size();	
		ArrayList<SubsetDiscreteData> discreteData= getDiscreteLists(trainingDataList, i);	
		double iGain = entropyCalculation(trainingDataList);	
		Iterator it = discreteData.iterator();
		while (it.hasNext()) {
			SubsetDiscreteData element = (SubsetDiscreteData) it.next();
			ArrayList<Data> list =element.getData();
			double listSize = list.size();
			double entropy=entropyCalculation(list);
			iGain -= (listSize/n)*entropy;	
		}
		return iGain;
	}


	/**constructTreeUsingMisclassification METHOD:Constructing the tree with the  mISCLASSIFICATION Impurity Calculation and Information Gain and 
	 * Selecting the attribute which has the highest Information Gain as the root node  and the iterating through its possible values(a,g,c,t)
	 * and calling the method recursively to builds the tree.
	 * @param trainingDataList
	 * @param parenNode
	 * @param csv
	 * @return node(Constructed Tree)
	 */
	public Node constructTreeUsingMisclassification(
			ArrayList<Data> trainingDataList, Node parenNode, double csv) {
		// TODO Auto-generated method stub
				double entropyValue;
				entropyValue = misClassification(trainingDataList);	
				if (entropyValue == 0) {						
					parenNode.setClassType(trainingDataList.get(0).getClassType());	
					return parenNode;									
				}else{
					int attributeNumber = highInformationgainAttributewithMisClassification(trainingDataList);	
					if (getThreshold(trainingDataList,attributeNumber) > csv) {
						ArrayList<SubsetDiscreteData> subset = getDiscreteLists(trainingDataList, attributeNumber);
						Iterator it1 = subset.iterator();
						while (it1.hasNext()) {	
							SubsetDiscreteData element = (SubsetDiscreteData) it1.next();
							ArrayList<Data> list =element.getData();
							Node childNode = new Node();
							childNode.setAttributeNumber(attributeNumber);
							childNode.setAttributeValue(element.getValue());
							parenNode.addNextNodeData(childNode);			
							constructTreeUsingInformatonGain(list,childNode,csv);		
						}
					}else{	
						parenNode.setClassType(setMaxtoLeaf(trainingDataList));	
						return parenNode;												}
				}
				
				return parenNode;
	}
	/**highInformationgainAttributewithMisClassification METHOD:Used to Identify the Attribute Id with Highest Information Gain to set it as a Root NODE
	 * @param trainingDataList
	 * @return aTTRIBUTE nUMBER with Highest Information Gain
	 */
	private int highInformationgainAttributewithMisClassification(
			ArrayList<Data> trainingDataList) {
		// TODO Auto-generated method stub
		double infomationGain=-9999999;				
		int attributeNodePosition=0;	
		boolean a=true;
		char[] sdata = trainingDataList.get(0).getAttributeValue();	
		for(int i=0;i<sdata.length;i++) {						
			if (a) {
				attributeNodePosition=i;
				a = false;
			}
			double infoGain = informationGainCalculationwithMisClassification(trainingDataList,i);
			if (infoGain>infomationGain) {							
				infomationGain = infoGain;
				attributeNodePosition = i;					
			}
		}
		return attributeNodePosition;
	}

	/**informationGainCalculationwithMisClassification METHOD: Gain(S, A) = Entropy(S) - S ((|Sv| / |S|) * Entropy(Sv)) is calculated using this method
	 * @param trainingDataList
	 * @param i-Attribute Id
	 * @return Information Gain
	 */
	private double informationGainCalculationwithMisClassification(
			ArrayList<Data> trainingDataList, int i) {
		// TODO Auto-generated method stub
		ArrayList<SubsetDiscreteData> discreteData= getDiscreteLists(trainingDataList, i);
		double n = trainingDataList.size();		
		double iGain = misClassification(trainingDataList);	
		Iterator it = discreteData.iterator();
		while (it.hasNext()) {
			SubsetDiscreteData element = (SubsetDiscreteData) it.next();
			ArrayList<Data> list =element.getData();
			double listSize = list.size();
			double entropy=misClassification(list);
			iGain -= (listSize/n)*entropy;	
		}
		return iGain;
	}

	/**misClassification METHOD:Calculation of Misclassification Impurity using formula with Promoter Count and Non Promoter Count
	 * @param trainingDataList
	 * @return double misClassification
	 */
	private double misClassification(ArrayList<Data> trainingDataList) {
		// TODO Auto-generated method stub
		Map<String, List<Data>> m1 = getMapData(trainingDataList);		
		double promoter = m1.get("Promoter").size();
		double nonPromoter = m1.get("Non-Promoter").size();	
		if (promoter > nonPromoter) {
        double nonpr=nonPromoter / (promoter+nonPromoter);
			return nonpr ;
		}else{
			double prcount=promoter / (promoter+nonPromoter);
			return prcount;
		}
	}
}
	