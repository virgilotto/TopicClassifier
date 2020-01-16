
import weka.core.*;
import weka.core.FastVector;
import weka.classifiers.meta.FilteredClassifier;

import java.io.*;


public class Classifier {


    String text;
    Instances instances;
    FilteredClassifier classifier;

    public void load(String fileName) {
        text = fileName;
        System.out.println("===== Loaded text data: " + fileName + " =====");
        System.out.println(text);
    }

    public void loadModel(String fileName) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            System.out.println("===== Loaded model: " + fileName + " =====");
        }
        catch (Exception e) {
            System.out.println("Problem found when reading: " + fileName);
        }
    }

    public void makeInstance() {
        FastVector fvNominalVal = new FastVector(5);
        fvNominalVal.addElement("business");
        fvNominalVal.addElement("entertainment");
        fvNominalVal.addElement("politics");
        fvNominalVal.addElement("sport");
        fvNominalVal.addElement("tech");
        Attribute attribute1 = new Attribute("class", fvNominalVal);
        Attribute attribute2 = new Attribute("text",(FastVector) null);
        FastVector fvWekaAttributes = new FastVector(2);
        fvWekaAttributes.addElement(attribute1);
        fvWekaAttributes.addElement(attribute2);
        instances = new Instances("Test relation", fvWekaAttributes, 1);
        instances.setClassIndex(0);
        DenseInstance instance = new DenseInstance(2);
        instance.setValue(attribute2, text);
        instances.add(instance);
        System.out.println("===== Instance created with reference dataset =====");
        System.out.println(instances);
    }

    public String classify() throws Exception {
        double pred = classifier.classifyInstance(instances.instance(0));
        System.out.println("===== Classified instance =====");
        System.out.println("Class predicted: " + instances.classAttribute().value((int) pred));

        return instances.classAttribute().value((int) pred);
    }
}	