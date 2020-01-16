
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader.ArffReader;
import java.io.*;

public class Learner {

    Instances trainData;
    StringToWordVector filter;
    FilteredClassifier classifier;

    public void loadDataset(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            ArffReader arff = new ArffReader(reader);
            trainData = arff.getData();
            System.out.println("===== Loaded dataset: " + fileName + " =====");
            reader.close();
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileName);
        }
    }

    public void evaluate() {
        try {
            trainData.setClassIndex(0);
            filter = new StringToWordVector();
            filter.setAttributeIndices("last");
            classifier = new FilteredClassifier();
            classifier.setFilter(filter);
            classifier.setClassifier(new NaiveBayes());
            Evaluation eval = new Evaluation(trainData);
            eval.crossValidateModel(classifier, trainData, 4, new Random(1));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println("===== Evaluating on filtered (training) dataset done =====");
        }
        catch (Exception e) {
            System.out.println("Problem found when evaluating");
        }
    }

    public void learn() {
        try {
            trainData.setClassIndex(0);
            filter = new StringToWordVector();
            filter.setAttributeIndices("last");
            classifier = new FilteredClassifier();
            classifier.setFilter(filter);
            classifier.setClassifier(new NaiveBayes());
            classifier.buildClassifier(trainData);
            // Uncomment to see the classifier
            // System.out.println(classifier);
            System.out.println("===== Training on filtered (training) dataset done =====");
        }
        catch (Exception e) {
            System.out.println("Problem found when training");
        }
    }

    public void saveModel(String fileName) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
            System.out.println("===== Saved model: " + fileName + " =====");
        }
        catch (IOException e) {
            System.out.println("Problem found when writing: " + fileName);
        }
    }

    public static void main (String[] args) {

        Learner learner;
        learner = new Learner();
        learner.loadDataset("topic.arff");
        learner.evaluate();
        learner.learn();
        learner.saveModel("myClassifier.dat");
    }
}