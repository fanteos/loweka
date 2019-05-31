package weka.classifiers.lolita;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import lolita.fuzo.aggregations.Aggregation;
import lolita.fuzo.aggregations.AggregationNorm;
import lolita.fuzo.aggregations.ArithmeticMean;
import lolita.fuzo.aggregations.COWA;
import lolita.fuzo.aggregations.Median;
import lolita.fuzo.aggregations.OverlapAggregation;
import lolita.fuzo.mixtures.GeneralizedMixtureFunction;
import lolita.fuzo.mixtures.QuasiUniversalMixtureFunction;
import lolita.fuzo.norms.AczelTnorm;
import lolita.fuzo.norms.DombiTnorm;
import lolita.fuzo.norms.DrasticTnorm;
import lolita.fuzo.norms.FrankTnorm;
import lolita.fuzo.norms.GodelTconorm;
import lolita.fuzo.norms.GodelTnorm;
import lolita.fuzo.norms.HamacherTnorm;
import lolita.fuzo.norms.LukasiewiczTnorm;
import lolita.fuzo.norms.NilpotentMinimumTnorm;
import lolita.fuzo.norms.ProductTnorm;
import lolita.fuzo.overlap.OverlapDB;
import lolita.fuzo.overlap.OverlapMM;
import lolita.fuzo.overlap.OverlapPowP;
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

public class CEMFPlus extends ParallelMultipleClassifiersCombiner implements TechnicalInformationHandler {

	private static final long serialVersionUID = 14001212L;
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

	/** combination rule: H-median */
	protected static final int H_MEDIAN_B = 6;
	/** combination rule: H-max */
	protected static final int H_MAX_B = 7;
	/** combination rule: H-average */
	protected static final int H_AVERAGE_B = 8;
	/** combination rule: H-product */
	protected static final int H_PRODUCT_B = 9;
	/** combination rule: H-min */
	protected static final int H_MIN_B = 10;
	/** combination rule: Overlap mM */
	protected static final int H_OVER_MM = 11;
	/** combination rule: Overlap pow p */
	protected static final int H_OVER_POW_P = 12;
	/** combination rule: Overlap DB */
	protected static final int H_OVER_DB = 13;
	/** combination rule: Lukasiewicz t-norm */
	protected static final int H_LUKAS = 14;
	/** combination rule: Drastic t-norm */
	protected static final int H_DRA = 15;
	/** combination rule: Nilpotent minimun t-norm */
	protected static final int H_NIL = 16;
	/** combination rule: Aczel-Alsina t-norm */
	protected static final int H_ACZ = 17;
	/** combination rule: Dombi t-norm */
	protected static final int H_DOM = 18;
	/** combination rule: Frank t-norm */
	protected static final int H_FRA = 19;
	/** combination rule: Hamacher t-norm */
	protected static final int H_HAM = 20;

	protected static final Tag[] TAGS_RULES = { new Tag(H_MEDIAN, "Med", "Median"), new Tag(H_MAX, "Max", "Max"),
			new Tag(H_AVERAGE, "Ari", "Arithmetic mean"), new Tag(H_PRODUCT, "Pro", "Product"),
			new Tag(H_MIN, "Min", "Minimun"), new Tag(H_cOWA, "H-cOWA", "cOWA") };

	protected static final Tag[] TAGS_RULESB = { new Tag(H_MEDIAN_B, "Med", "Median"), new Tag(H_MAX_B, "Max", "Max"),
			new Tag(H_AVERAGE_B, "Mean", "Arithmetic mean"), new Tag(H_PRODUCT_B, "Pro", "Product"),
			new Tag(H_MIN_B, "Min", "Minimun"), new Tag(H_OVER_MM, "OnM", "OverlapMM"), new Tag(H_OVER_POW_P, "Op", "OverlapPowP"),
			new Tag(H_OVER_DB, "Odb", "OverlapDB"),  new Tag(H_LUKAS, "L-tnorm", "Lukasiewicz t-norm"), 
			new Tag(H_DRA, "D-tnorm", "Drastic t-norm"), new Tag(H_NIL, "Nil-tnorm", "Nilpotent min t-norm"),
			new Tag(H_ACZ, "A-tnorm", "Aczel t-norm"), new Tag(H_DOM, "Do-tnorm", "Dombi t-norm"), 
			new Tag(H_FRA, "F-tnorm", "Frank t-norm"), new Tag(H_HAM, "H-tnorm", "Hamacher t-norm")};

	/** Number of folds */
	protected int m_NumFolds = 10;

	/** Combination Rule variable */
	protected int m_ReferentialFunction = H_MEDIAN;

	/** Combination Rule variable B */
	protected int m_PseudoConjunction = H_MEDIAN_B;

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>(1);
		result.addElement(new Option("\tThe first mixture function to use\n" + "\t(default: H-Med)", "A", 1,
				"-A " + Tag.toOptionList(TAGS_RULES)));
		result.addElement(new Option("\tThe second mixture function to use\n" + "\t(default: H-Med)", "B", 1,
				"-B " + Tag.toOptionList(TAGS_RULESB)));
		result.addElement(new Option("\tSets the number of cross-validation folds.", "X", 1, "-X <number of folds>"));
		result.addAll(Collections.list(super.listOptions()));
		return result.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		String tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0) {
			setReferentialFunction(new SelectedTag(tmpStr, TAGS_RULES));
		} else {
			setReferentialFunction(new SelectedTag(H_MEDIAN, TAGS_RULES));
		}
		String tmpStrB = Utils.getOption('B', options);
		if (tmpStrB.length() != 0) {
			setPseudoConjunction(new SelectedTag(tmpStrB, TAGS_RULESB));
		} else {
			setPseudoConjunction(new SelectedTag(H_MEDIAN_B, TAGS_RULESB));
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
		String[] options = new String[superOptions.length + 6];

		int current = 0;
		options[current++] = "-A";
		options[current++] = "" + getReferentialFunction();
		options[current++] = "-B";
		options[current++] = "" + getPseudoConjunction();
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
	 * Gets the combination rule used
	 * 
	 * @return the combination rule used
	 */
	public SelectedTag getPseudoConjunction() {
		return new SelectedTag(m_PseudoConjunction, TAGS_RULESB);
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
	 * Sets the combination rule to use. Values other than
	 * 
	 * @param newRule the combination rule method to use
	 */
	public void setPseudoConjunction(SelectedTag newRuleB) {
		if (newRuleB.getTags() == TAGS_RULESB) {
			m_PseudoConjunction = newRuleB.getSelectedTag().getID();
		}
	}

	public String pseudoConjunctionTipText() {
		return "The function that replaces the product in the mixture function, can be seen as pseudo conjunction.";
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
			throw new IllegalArgumentException("The Number of cross-validation folds must be positive.");
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
		QuasiUniversalMixtureFunction GM = createMixtureFunction();
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
		double sum = 0.0;
		for(int i = 0; i < distribution.length; i++) {
			sum = sum + distribution[i];
		}
		if(sum != 0.0) {
			Utils.normalize(distribution);
		}
		return distribution;
	}

	protected QuasiUniversalMixtureFunction createMixtureFunction() {
		QuasiUniversalMixtureFunction GM;
		Aggregation A;
		Aggregation B;
		switch (m_ReferentialFunction) {
			case H_MEDIAN:
				A = new Median();
				break;
			case H_MAX:
				A = new AggregationNorm(new GodelTconorm());
				break;
			case H_AVERAGE:
				A = new ArithmeticMean();
				break;
			case H_MIN:
				A = new AggregationNorm(new GodelTnorm());
				break;
			case H_PRODUCT:
				A = new AggregationNorm(new ProductTnorm());
				break;
			case H_cOWA:
				A = new COWA();
				break;
			default:
				A = null;
				break;

		}
		
		switch(m_PseudoConjunction) {
			case H_MEDIAN_B:
				B = new Median();
				break;
			case H_MAX_B:
				B = new AggregationNorm(new GodelTconorm());
				break;
			case H_AVERAGE_B:
				B = new ArithmeticMean();
				break;
			case H_MIN_B:
				B = new AggregationNorm(new GodelTnorm());
				break;
			case H_PRODUCT_B:
				B =  new AggregationNorm(new ProductTnorm());
				break;
			case H_OVER_MM:
				B = new OverlapAggregation(new OverlapMM());
				break;
			case H_OVER_POW_P:
				B = new OverlapAggregation(new OverlapPowP(m_Classifiers.length));
				break;
			case H_OVER_DB:
				B = new OverlapAggregation(new OverlapDB());
				break;
			case H_LUKAS:
				B =  new AggregationNorm(new LukasiewiczTnorm());
				break;
			case H_DRA:
				B =  new AggregationNorm(new DrasticTnorm());
				break;
			case H_NIL:
				B = new AggregationNorm(new NilpotentMinimumTnorm());
				break;
			case H_ACZ:
				B = new AggregationNorm(new AczelTnorm());
				break;
			case H_DOM:
				B = new AggregationNorm(new DombiTnorm());
				break;
			case H_FRA:
				B = new AggregationNorm(new FrankTnorm());
				break;
			case H_HAM:
				B = new AggregationNorm(new HamacherTnorm());
				break;
			default:
				B = null;
				break;
		}
		if(A != null && B != null) {
			GM = new QuasiUniversalMixtureFunction(A, B);
			return GM;
		}else {
			return null;
		}
	}
	
	protected double[] generateVector(double[][] matrix, int col){
		double[] vetor = new double[this.m_Classifiers.length];
		for(int i = 0; i < m_Classifiers.length; i++) {
			vetor[i] = matrix[i][col];
		}
		return vetor;	
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Valdigleis S. Costa et al.");
		result.setValue(Field.YEAR, "2019");
		result.setValue(Field.TITLE, "---");
		result.setValue(Field.JOURNAL, "---");
		result.setValue(Field.VOLUME, "---");
		result.setValue(Field.PAGES, "---");
		return null;
	}
}
