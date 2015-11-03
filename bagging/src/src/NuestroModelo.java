package src;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;

import distance.*;
import weka.core.Instance;
import weka.core.Instances;

public class NuestroModelo implements Modelo { 
	
	private double[][] lista;
    /**
     * 1- Numero de vecinos para analizar
     * */
	protected int kNN;
	/**
	 * 1- "No distance weighting"
	 * 2- "Weight by 1/distance"
	 */
	public enum DistanceWight {
		NoDistance, OneDivDistance
	};
	
	protected DistanceWight distanceWeighting;
	
	private int distance;
	private Minkowski distanceMethod;
	
	
	private int[][] matrizConf;
	private Enumeration<Object> clases;
	private Instances instanciasBuenas;
	
	/**
	 * Constructor para crear el modelo KNN.
	 * @param KNN numero de vecionos a analizar. [1:]
	 * @param distance Tipo de distancia a analizar: [(1: Manhattan), (2: Euclídea), (3: Minkowski)]
	 * @param searchAlgoritm Algoritmo de busqueda: [1:5]
	 */
	public NuestroModelo(int KNN, DistanceWight distanceW, int distanceT){
	    this.setKNN(KNN);
	    this.setDistanceWeighting(distanceW);
	    this.setDistance(distanceT);
	}
	
	public int getDistance() {
		return distance;
	}

	public void setDistance(int distance) {
		this.distance = distance;
		if (distance >= 1){
			this.distanceMethod = new Minkowski(distance);
		}else{
			this.distanceMethod = new Minkowski(1);
		}
	}

	public void setDistanceWeighting(DistanceWight i){
		this.distanceWeighting=i;
		
	}
	public DistanceWight getDistanceWeighting(){
		return this.distanceWeighting;
	}

	public int getKNN(){
		return kNN;
	}

	public void setKNN(int k) {
		if (k<1) k = 1;
		this.kNN=k;
	} 
   	
	public void crearMatrixConfusion(Instances noclasf, Instances clasf){
		this.clases = noclasf.classAttribute().enumerateValues();
		this.matrizConf = new int[noclasf.classAttribute().numValues()][noclasf.classAttribute().numValues()];
		for (int f = 0; f < this.matrizConf.length; f++) {
			for (int i = 0; i < this.matrizConf.length; i++) {
				this.matrizConf[f][i] = 0;
			}
		}
		for (int i = 0; i < clasf.numInstances(); i++) {
			int claseNC = (int) noclasf.get(i).classValue();
			int claseC = (int) clasf.get(i).classValue();
			this.matrizConf[claseC][claseNC]++;
		}
	}
	public double precision(int classIndex) {

	    double correct = 0, total = 0;
	    for (int i = 0; i < this.matrizConf.length; i++) {
	      if (i == classIndex) {
	        correct += this.matrizConf[i][classIndex];
	      }
	      total += this.matrizConf[i][classIndex];
	    }
	    if (total == 0) {
	      return 0;
	    }
	    return correct / total;
	  }
	
	public double weightedPrecision() {
	    double[] classCounts = new double[this.matrizConf.length];
	    double classCountSum = 0;

	    for (int i = 0; i < this.matrizConf.length; i++) {
	      for (int j = 0; j < this.matrizConf.length; j++) {
	        classCounts[i] += this.matrizConf[i][j];
	      }
	      classCountSum += classCounts[i];
	    }

	    double precisionTotal = 0;
	    for (int i = 0; i < this.matrizConf.length; i++) {
	      double temp = this.precision(i);
	      precisionTotal += (temp * classCounts[i]);
	    }

	    return precisionTotal / classCountSum;
	  }
	
	public double recall(int classIndex) {

	    double correct = 0, total = 0;
	    for (int j = 0; j < this.matrizConf.length; j++) {
	      if (j == classIndex) {
	        correct += this.matrizConf[classIndex][j];
	      }
	      total += this.matrizConf[classIndex][j];
	    }
	    if (total == 0) {
	      return 0;
	    }
	    return correct / total;
	}
	
	public double weightedRecall() {
	    double[] classCounts = new double[this.matrizConf.length];
	    double classCountSum = 0;

	    for (int i = 0; i < this.matrizConf.length; i++) {
	      for (int j = 0; j < this.matrizConf.length; j++) {
	        classCounts[i] += this.matrizConf[i][j];
	      }
	      classCountSum += classCounts[i];
	    }

	    double truePosTotal = 0;
	    for (int i = 0; i < this.matrizConf.length; i++) {
	    	double temp = this.recall(i);
	      truePosTotal += (temp * classCounts[i]);
	    }

	    return truePosTotal / classCountSum;
	  }
	
	public double accuracy(){
		double count=0;
		double correct=0;
		for (int i = 0; i < this.matrizConf.length; i++) {
			for (int j = 0; j < this.matrizConf.length; j++) {
				if (i==j) correct += this.matrizConf[i][j];
				count+=this.matrizConf[i][j];
			}
		}
		return correct/count;
	}
	
	public String calcularMediciones(){
		double recall = this.weightedRecall();
		double accuracy = this.accuracy();
		double precision = this.weightedPrecision();
		
		String result = "";
		result += ("\n**************************************\n");
		result += ("\n****Estimacion Nuestro modelo k-NN****\n");
		result += ("\n**************************************\n");
		result += ("K: " + kNN+"\n");
		result += ("Distance Weighting: " + this.getDistanceWeighting()+"\n");
		result += ("Distance Type (Minkowski): " + this.getDistance()+"\n");
		result += ("+------------------------------------+\n");
		result += ("|   Precision = "+precision+"   |\n");
		result += ("|   Recall    = "+recall+"   |\n");
		result += ("|   Accuracy  = "+accuracy+"   |\n");
		result += ("+------------------------------------+\n");
		result += ("------------Matriz De Confusión-------\n");
		result += ("\n");
		result += "\t";
		for (int i = 0; i < this.matrizConf.length; i++) {
			result +=("+-------");
		}
		result+="+\n";
		result += "\t";
		for (int i = 0; i < this.matrizConf.length; i++) {
			result +=("| "+(char)(97+i) + "\t");
		}
		result+="|  <-- Clasificado como\n";
		result += "\t";
		for (int i = 0; i < this.matrizConf.length; i++) {
			result +=("+-------");
		}
		result+="+\n";
		for (int f = 0; f < this.matrizConf.length; f++) {
			result += "\t";
			for (int i = 0; i < this.matrizConf.length; i++) {
				result +=("| "+this.matrizConf[f][i] + "\t");
			}
			result+="| "+(char)(97+f)+" = "+this.clases.nextElement()+"\n";
			result += "\t";
			for (int i = 0; i < this.matrizConf.length; i++) {
				result +=("+-------");
			}
			result+="+\n";
		}
		return result;
	}
	private double calcularPeso(double distancia){
		
		switch (this.getDistanceWeighting()) {
		case NoDistance:
			return distancia;
		case OneDivDistance:
			return 1.0/distancia;
		}
		return distancia;
	}

	public double clasificarInstancia(Instance NoClasificada){
		int numerovecinos = this.getKNN();
		//recorreremos el array hasta el numero de k en instancias
		double[] mediasPeso = new double[this.instanciasBuenas.numClasses()];
		Double[][] temp= new Double[this.instanciasBuenas.numClasses()][2];
		for(int i=0;i<temp.length;i++){
			for (int j = 0; j < temp[i].length; j++) {
				temp[i][j]=0.00;
			}
		}
		for(int i=0;i<numerovecinos;i++){
			temp[(int)this.instanciasBuenas.get((int)lista[i][0]).classValue()][0]+=calcularPeso(lista[i][1]);
			temp[(int)this.instanciasBuenas.get((int)lista[i][0]).classValue()][1]++;
		}
		for(int i=0;i<this.instanciasBuenas.numClasses();i++){
			if(temp[i][1]>0){
				mediasPeso[i]=temp[i][0]/temp[i][1];
			}else{
				mediasPeso[i]=-1;
			}
		}
		return conseguirClase(mediasPeso);
	}

	private double conseguirClase(double[] mediasPeso) {
		int pos=-1;
		double peso=Double.MAX_VALUE;
		switch (this.getDistanceWeighting()) {
		case NoDistance:
			for(int i=0;i<mediasPeso.length;i++){					
				if(mediasPeso[i]<peso && mediasPeso[i]!=-1){
					pos = i;
					peso = mediasPeso[i];
				}
			}
			break;
		case OneDivDistance:
			peso=Double.MIN_VALUE;
			for(int i=0;i<mediasPeso.length;i++){					
				if(mediasPeso[i]>peso && mediasPeso[i]!=-1){
					pos = i;
					peso = mediasPeso[i];
				}
			}
			break;
		}
		return pos;
		
	}
	public void buildClasifier(Instances instancias){
		this.instanciasBuenas = instancias;
	}

	public void prepararInstancias(Instance instancia) {
		double distancia = 0.00;
		lista = new double[this.instanciasBuenas.numInstances()][2];
		
		//primera linea referencia instancia; segunda linea distancia
		for(int j=0;j<this.instanciasBuenas.numInstances();j++){
			distancia = this.distanceMethod.calcularDistancia(this.instanciasBuenas.get(j), instancia);
			lista[j][0]=j;
			lista[j][1]=distancia;
		}
		Collections.sort(Arrays.asList(lista), new Comparator<double[]>() {

			@Override
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[1], o2[1]);
			}
		});
		
	}

	@Override
	public void evaluarModelo(Instances instanciasAEvaluar) {
		Instances copia = new Instances(instanciasAEvaluar, 0, instanciasAEvaluar.numInstances());
		for (int i=0;i<copia.numInstances();i++) {
			this.prepararInstancias(copia.get(i));
			copia.get(i).setClassValue(this.clasificarInstancia(copia.get(i)));
		}
		this.crearMatrixConfusion(copia, instanciasAEvaluar);
	}
}	
