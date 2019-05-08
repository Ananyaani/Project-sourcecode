
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.core.Attribute;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.lazy.IBk;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.neighboursearch.KDTree;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class DecisionTreeClassifier {
    
     Classifier cModel;
   
    

    Attribute per = new Attribute("period");
   
    Attribute s1 = new Attribute("symptom1");
    Attribute s2 = new Attribute("symptom2");
    Attribute s3 = new Attribute("symptom3");
    Attribute s4 = new Attribute("symptom4");
    
    
   
    
    FastVector fvClassVal = new FastVector(2);
    Attribute ClassAttribute;
    FastVector fvWekaAttributes = new FastVector(5);
    Instances isTrainingSet;
    
    
    
    
    DecisionTreeClassifier()
    {
        
        fvClassVal.addElement("1");
         fvClassVal.addElement("0");
         ClassAttribute = new Attribute("theClass", fvClassVal);

        
         fvWekaAttributes.addElement(per);
        
         fvWekaAttributes.addElement(s1);
         fvWekaAttributes.addElement(s2);
         fvWekaAttributes.addElement(s3);
         fvWekaAttributes.addElement(s4);
         
         fvWekaAttributes.addElement(ClassAttribute);
         
        
        
        
        
        
    }
    
    
       int getSeverity(String line)
    {
        String [] parts = line.split(",");
        
        double v = Double.parseDouble(parts[1]);
        
        
        double v1 = Double.parseDouble(parts[2]);
        
        double v2 = Double.parseDouble(parts[3]);
        
        double v3 = Double.parseDouble(parts[4]);
        
        if (((v>=1) && (v <= 3)) || ((v1>=1) && (v1 <= 3)) || ((v2>=1) && (v2 <= 3)) || ((v3>=1) && (v3 <= 3)))
        {
            return 1; //"Jan-Feb";
        }        
        else if (((v>3) && (v <= 6)) || ((v1>3) && (v1 <= 6)) || ((v2>3) && (v2 <= 6)) || ((v3>3) && (v3 <= 6)))
        {
            return 2; //"Mar-May";
        }
       else if (((v>6) && (v <= 9)) || ((v1>6) && (v1 <= 9)) || ((v2>6) && (v2 <= 9)) || ((v3>6) && (v3 <= 9)))
        {
            return 3; //"May-Sep";
        }
          else if ((v>9) && (v <= 12) || (v1>9) && (v1 <= 12) || (v2>9) && (v2 <= 12) || (v3>9) && (v3 <= 12))
        {
            return 4; //"Oct-Dec";
        }
        return 0; //"AVERAGE";
        
    }
     
      int getdislevel(String line)
    {
        String [] parts = line.split(",");
        
        double v = Double.parseDouble(parts[1]);
        
        double v1 = Double.parseDouble(parts[2]);
        
        double v2 = Double.parseDouble(parts[3]);
        
        double v3 = Double.parseDouble(parts[4]);
        
        if ((v>=1) && (v <= 3))
        {
            return 1; //"";
        }   
        
        else if((v1>=1) && (v1 <= 3))
        {
            return 2; //"";
        }
        else if((v2>=1) && (v2 <= 3))
        {
            return 3; //"";
        }
        else if((v3>=1) && (v3 <= 3))
        {
            return 4; //"";
        }
        else if ((v>3) && (v <= 6))
        {
            return 5; //"";
        }
        else if((v1>3) && (v1 <= 6))
        {
            return 6; //"";
        }
        else if((v2>3) && (v2 <= 6))
        {
            return 7; //"";
        }
        else if((v3>3) && (v3 <= 6))
        {
            return 8; //"";
        }
        
       else if ((v>6) && (v <= 9))
        {
            return 9; //"";
        }
       else if((v1>6) && (v1 <= 9))
               {
                 return 10; //"";  
               }
       else if((v2>6) && (v2 <= 9))
               {
                   return 11; //"";
               }
       else if((v3>6) && (v3 <= 9))
               {
                   return 12; //"";
               }
          else if ((v>9) && (v <= 12))
        {
            return 13; //"";
        }
          else if((v1>9) && (v1 <= 12))
                  {
                    return 14; //"";  
                  }
          else if((v2>9) && (v2 <= 12))
                  {
                     return 15; //""; 
                  }
          else if((v3>9) && (v3 <= 12))
                  {
                      return 16; //"";
                  }
        return 0; //"AVERAGE";
        
    }
     
     
     
public void trainClassifierDT()
    {


        int count = 0;
        
        try
        {
           FileReader fr = new FileReader("preprocessedtrain.txt");

           BufferedReader buf=new BufferedReader(fr);

           String s;

           while ((s=buf.readLine())!=null)
           {
                 if (s.length()<2)
                 {
                     continue;
                 }
                 count++;
           }
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
       
        
        isTrainingSet = new Instances("Rel", fvWekaAttributes, count);
        // Set class index
        isTrainingSet.setClassIndex(5);
         
        try
        {
           FileReader fr = new FileReader("preprocessedtrain.txt");

           BufferedReader buf=new BufferedReader(fr);

           String s;

           while ((s=buf.readLine())!=null)
           {
                 if (s.length()<2)
                 {
                     continue;
                 }
                
                 System.out.println(s);
                 Instance iExample = new Instance(6);
                 
                 String [] parts = s.split(",");
            
                
                

                iExample.setValue((Attribute)fvWekaAttributes.elementAt(0),Integer.parseInt(parts[0]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(1),Integer.parseInt(parts[1]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(2),Integer.parseInt(parts[2]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(3),Integer.parseInt(parts[3]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(4),Integer.parseInt(parts[4]));
              
               
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(5),parts[5]);
                isTrainingSet.add(iExample);
                
                
                
             
               
           }
           buf.close();
           fr.close();
           
        
           
           cModel = new J48(); // Decision Tree Classifier Module
           
            cModel.buildClassifier(isTrainingSet);
            
                       
         
            
           
           
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }



     
     
      
      
    boolean isDiseaseDT(String line)
    {
        
        String [] parts = line.split(",");
        
        Instance iExample = new Instance(6);
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(0),Integer.parseInt(parts[0]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(1),Integer.parseInt(parts[1]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(2),Integer.parseInt(parts[2]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(3),Integer.parseInt(parts[3]));
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(4),Integer.parseInt(parts[4]));
               
                //iExample.setValue((Attribute)fvWekaAttributes.elementAt(8),Integer.parseInt(parts[8]));
                //iExample.setValue((Attribute)fvWekaAttributes.elementAt(24),parts[24]);
        
        
               
        
        
        iExample.setDataset(isTrainingSet);
        
             try
             {
                double[] fDistribution = cModel.distributionForInstance(iExample);
                 
                 
                
                 
                 double bigval0 = fDistribution[0];
                 double bigval1 = fDistribution[1];
                 
                 
                 
                 
                 
                 if (bigval0 >bigval1)
                 {
                     return true;
                 }
                 
            
             }
             catch(Exception e)
             {
                 e.printStackTrace();
             }
        
             return false;
        
        
    }
    
   
    
    
    
    
    public static void main(String []arg)
    {
        DecisionTreeClassifier DT = new DecisionTreeClassifier();

        try
        {
            DT.trainClassifierDT();
            
            System.out.println("Training completed Decision Tree!!!");
            
            
           
            
            String ct = "1,1,1,2,3,4";
            
            boolean res=DT.isDiseaseDT(ct);
            
            System.out.println("Returned result for Decision tree:" + res);
            
           
          
            
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        
        
        
        
    }
    
    
    
    
}
