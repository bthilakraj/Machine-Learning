package com.ml.decisionTree;

import java.util.ArrayList;

/**
 * @author Thilak
 * SubsetData Class :  Subset generated with repect to a g c t Used to structure the subset storing the dna value (agcta ... to a char[57] of attributeValue 
 *and ClassTpe if its promoter or Non Promoter
 */
public class SubsetDiscreteData {
String value;
ArrayList<Data> data;
public String getValue() {
	return value;
}
public void setValue(String value) {
	this.value = value;
}
public ArrayList<Data> getData() {
	return data;
}
public void setData(ArrayList<Data> data) {
	this.data = data;
}

}
