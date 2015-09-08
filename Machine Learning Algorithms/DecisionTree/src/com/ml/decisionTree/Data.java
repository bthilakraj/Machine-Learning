package com.ml.decisionTree;


import java.util.Map;

/**
 * @author Thilak
 *Data Class:  Used to structure the input file read from Text File and storing the dna value (agcta ... to a char[57] of attributeValue 
 *and ClassTpe if its promoter or Non Promoter
 */
public class Data {
	private char[] attributeValue;
	private String classType;
	
	public char[] getAttributeValue() {
		return attributeValue;
	}
	public void setAttributeValue(char[] attributeValue) {
		this.attributeValue = attributeValue;
	}
	public String getClassType() {
		return classType;
	}
	public void setClassType(String classType) {
		this.classType = classType;
	}
	public char getAttributeValue(int key) {
		return attributeValue[key];
	}
	
	
}
