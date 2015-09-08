package com.ml.decisionTree;

import java.util.ArrayList;
import java.util.List;



/**
 * @author Thilak
 * Node Class Datastructure which holds the constructed tree data where node denotes the next 
 * node if available,Attributenumber,Value of the attribute(a,g,c,t) and Classtype(Promoter,Non Promoter)
 * It includes all getters setters
 */
public class Node {
	List<Node> node;	
	int attributeNumber;						
	String attributeValue;				
	String classType;
	public List<Node> getNode() {
		return node;
	}
	public void setNode(List<Node> node) {
		this.node = node;
	}
	public int getAttributeNumber() {
		return attributeNumber;
	}
	public void setAttributeNumber(int attributeNumber) {
		this.attributeNumber = attributeNumber;
	}
	public String getAttributeValue() {
		return attributeValue;
	}
	public void setAttributeValue(String attributeValue) {
		this.attributeValue = attributeValue;
	}
	public String getClassType() {
		return classType;
	}
	public void setClassType(String classType) {
		this.classType = classType;
	}
	public void addNextNodeData(Node nextNode) {
		this.node.add(nextNode);
	}
	public Node() {
		super();
		node = new ArrayList<Node>();
		this.attributeNumber = attributeNumber;
		this.attributeValue = attributeValue;
		classType = "X";
	}
}
