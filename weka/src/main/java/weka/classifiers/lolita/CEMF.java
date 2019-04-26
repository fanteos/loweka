package weka.classifiers.lolita;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import lolita.fuzo.aggregations.AggregationNorm;
import lolita.fuzo.aggregations.ArithmeticMean;
import lolita.fuzo.aggregations.COWA;
import lolita.fuzo.aggregations.Median;
import lolita.fuzo.mixtures.GeneralizedMixtureFunction;
import lolita.fuzo.norms.GodelTconorm;
import lolita.fuzo.norms.GodelTnorm;
import lolita.fuzo.norms.ProductTnorm;

import weka.classifiers.ParallelMultipleClassifiersCombiner;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

public class CEMF extends ParallelMultipleClassifiersCombiner implements TechnicalInformationHandler {

	private static final long serialVersionUID = 1431212L;
	/** combination rule: H-median */
	protected static final int H_MEDIAN = 1;
	/** combination rule: H-max */
	protected static final int H_MAX = 2;
	/** combination rule: H-average */
	protected static final int H_AVERAGE = 3;
	/** combination rule: H-product */
	protected static final int H_PRODUCT = 4;
	/** combination rule: H-min */
	protected static final int H_MIN = 5;
	/** combination rule: H-cOWA */
	protected static final int H_cOWA = 6;

	protected static final Tag[] TAGS_RULES = { new Tag(H_MEDIAN, "H-Med", "Median"), new Tag(H_MAX, "H-max", "Max"),
			new Tag(H_AVERAGE, "H-Ari", "Arithmetic mean"), new Tag(H_PRODUCT, "H-Pro", "Product"),
			new Tag(H_MIN, "H-Min", "Min"), new Tag(H_cOWA, "H-cOWA", "cOWA") };

	/** referential function variable */
	protected int m_ReferentialFunction = H_MEDIAN;

	/** Number of folds */
	protected int m_NumFolds = 10;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {
		return "Combines several classifiers using mixture functions. " + "Can do classification or regression.\n\n"
				+ "For more information, see\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Valdigleis S. Costa et al.");
		result.setValue(Field.YEAR, "2018");
		result.setValue(Field.TITLE,
				"Combining multiple algorithms in classifier ensembles using generalized mixture functions");
		result.setValue(Field.JOURNAL, "Neurocomputing");
		result.setValue(Field.VOLUME, "313");
		result.setValue(Field.PAGES, "402-414");
		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>(1);
		result.addElement(new Option("\tThe mixture function to use\n" + "\t(default: H-Med)", "R", 1,
				"-R " + Tag.toOptionList(TAGS_RULES)));
		result.addElement(new Option("\tSets the number of cross-validation folds.", "X", 1, "-X <number of folds>"));
		result.addAll(Collections.list(super.listOptions()));
		return result.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmpStr = Utils.getOption('R', options);
		if (tmpStr.length() != 0) {
			setReferentialFunction(new SelectedTag(tmpStr, TAGS_RULES));
		} else {
			setReferentialFunction(new SelectedTag(H_MEDIAN, TAGS_RULES));
		}
		String numFoldsString = Utils.getOption('X', options);
		if (numFoldsString.length() != 0) {
			setNumFolds(Integer.parseInt(numFoldsString));
		} else {
			setNumFolds(10);
		}
		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		String[] superOptions = super.getOptions();
		String[] options = new String[superOptions.length + 4];

		int current = 0;
		options[current++] = "-R";
		options[current++] = "" + getReferentialFunction();
		options[current++] = "-X";
		options[current++] = "" + getNumFolds();
		System.arraycopy(superOptions, 0, options, current, superOptions.length);
		return options;
	}

	/**
	 * Gets the combination rule used
	 * 
	 * @return the combination rule used
	 */
	public SelectedTag getReferentialFunction() {
		return new SelectedTag(m_ReferentialFunction, TAGS_RULES);
	}

	/**
	 * Sets the combination rule to use. Values other than
	 * 
	 * @param newRule the combination rule method to use
	 */
	public void setReferentialFunction(SelectedTag newRule) {
		if (newRule.getTags() == TAGS_RULES) {
			m_ReferentialFunction = newRule.getSelectedTag().getID();
		}
	}

	public String referentialFunctionTipText() {
		return "The referential function used to build the function H.";
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numFoldsTipText() {
		return "The number of folds used for cross-validation.";
	}

	/**
	 * Gets the number of folds for the cross-validation.
	 *
	 * @return the number of folds for the cross-validation
	 */
	public int getNumFolds() {
		return m_NumFolds;
	}

	/**
	 * Sets the number of folds for the cross-validation.
	 *
	 * @param numFolds the number of folds for the cross-validation
	 * @throws Exception if parameter illegal
	 */
	public void setNumFolds(int numFolds) throws Exception {
		if (numFolds < 0) {
			throw new IllegalArgumentException("The number of cross-validation folds must be positive.");
		}
		m_NumFolds = numFolds;
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result;
		result = super.getCapabilities();
		result.setMinimumNumberInstances(getNumFolds());
		return result;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances newData = new Instances(data);
		newData.deleteWithMissingClass();
		super.buildClassifier(newData);
		buildClassifiers(newData);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double result;
		double[] dist = distributionForInstance(instance);
		int index;
		if (instance.classAttribute().isNominal()) {
			index = Utils.maxIndex(dist);
			if (dist[index] == 0) {
				result = Utils.missingValue();
			} else {
				result = index;
			}
		} else if (instance.classAttribute().isNumeric()) {
			result = dist[0];
		} else {
			result = Utils.missingValue();
		}
		return result;
	}

	/**
	 * Returns class probabilities.
	 *
	 * @param instance the instance to be classified
	 * @return the distribution
	 * @throws Exception if instance could not be classified successfully
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// the distribution
		double[] distribution = new double[instance.numClasses()];
		// create matrix for results individual
		double[][] resultIndividual = new double[m_Classifiers.length][instance.numClasses()];
		// get each result individual
		for (int i = 0; i < m_Classifiers.length; i++) {
			double[] temp = m_Classifiers[i].distributionForInstance(instance);
			if (instance.classAttribute().isNumeric()) {
				resultIndividual[i][0] = temp[0];
			} else {
				for (int j = 0; j < instance.numClasses(); j++) {
					resultIndividual[i][j] = temp[j];
				}
			}

		}
		// Define the mixture function
		GeneralizedMixtureFunction GM = getMixtureFunction();
		// Apply the mixture function for obtain the distribution of ensemble
		if (instance.classAttribute().isNumeric()) {
			double[] v = generateVector(resultIndividual, 0);
			distribution[0] = GM.apply(v);
		} else {
			for (int i = 0; i < instance.numClasses(); i++) {
				double[] v = generateVector(resultIndividual, i);
				distribution[i] = GM.apply(v);
			}
		}
		Utils.normalize(distribution);
		return distribution;
	}

	protected GeneralizedMixtureFunction getMixtureFunction() {
		GeneralizedMixtureFunction g = null;
		switch (m_ReferentialFunction) {
		case H_MEDIAN:
			g = new GeneralizedMixtureFunction(new Median());
			break;
		case H_MAX:
			g = new GeneralizedMixtureFunction(new AggregationNorm(new GodelTconorm()));
			break;
		case H_AVERAGE:
			g = new GeneralizedMixtureFunction(new ArithmeticMean());
			break;
		case H_MIN:
			g = new GeneralizedMixtureFunction(new AggregationNorm(new GodelTnorm()));
			break;
		case H_PRODUCT:
			g = new GeneralizedMixtureFunction(new AggregationNorm(new ProductTnorm()));
			break;
		case H_cOWA:
			g = new GeneralizedMixtureFunction(new COWA());
			break;
		default:
			break;
		}
		return g;
	}

	protected double[] generateVector(double[][] matrix, int col) {
		double[] vetor = new double[m_Classifiers.length];
		for (int i = 0; i < m_Classifiers.length; i++) {
			vetor[i] = matrix[i][col];
		}
		return vetor;
	}
}
