package bag;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;

import src.Lector;
import src.Modelo;
import src.NuestroModelo;
import src.NuestroModelo.DistanceWight;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

public class Bagging implements Modelo {
	
	private int bagnum;
	private Instances misinstancias;
	private double accuracyMax;
	private int numclass;
	private int[][] matrizConf;
	private Modelo[] modelos;
	
	public Bagging(int l) {
		this.bagnum = l;
	}
	
	private Instances crearMuestras(){
		Instances instancias= new Instances(this.misinstancias,0);
		for (int i = 0; i < this.misinstancias.numInstances(); i++) {
			int random = (int)Math.floor(Math.random()*this.misinstancias.numInstances()); 
			instancias.add(this.misinstancias.get(random));
		}
		return instancias;
	}
	
	public static void main(String[] args) {
		Instances ins1 = Lector.getLector().leerInstancias("./breast-cancer.arff");
		Instances ins;
		Randomize ra= new Randomize();
		try {
			ra.setInputFormat(ins1);
			ins=Filter.useFilter(ins1, ra);
		} catch (Exception e) {
			ins=ins1;
			e.printStackTrace();
		}
		Instances[][] trozos = new Instances[10][2];
		int size =  ins.size()/10;
		double acumacc;
		double accuracyMax = 0.0;
		Bagging b;
		int numeroL = 0;
		for (int i = 2; i <= 15; i++) {
			acumacc = 0.0;
			b = new Bagging(i);
			for (int j = 1; j <= 10; j++) {
				trozos[j-1][0] = new Instances(ins, 0, size*(j-1));
				trozos[j-1][0].addAll(new Instances(ins, size*j, ins.size()-(size*j)));
				trozos[j-1][1] = new Instances(ins, size*(j-1), size);
			}
			for (int k = 0; k < 10; k++) {
				b.buildClasifier(trozos[k][0]);
				b.evaluarModelo(trozos[k][1]);
				System.out.println(b.calcularMediciones());
				acumacc+=b.accuracy();
			}
			acumacc=acumacc/10.0;
			System.out.println("L: "+i+" | Accuracy: "+acumacc);
			if(acumacc > accuracyMax){
				accuracyMax = acumacc;
				numeroL=i;
			}
		}
		System.out.println("\n\nBest L: "+numeroL+" | Accuracy: "+accuracyMax);
		

		
	}

	@Override
	public String calcularMediciones() {
		String result = "";
		Enumeration<Object> clases=this.misinstancias.classAttribute().enumerateValues();
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
			result+="| "+(char)(97+f)+" = "+clases.nextElement()+"\n";
			result += "\t";
			for (int i = 0; i < this.matrizConf.length; i++) {
				result +=("+-------");
			}
			result+="+\n";
		}
		return result;
	}

	@Override
	public double accuracy() {
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

	@Override
	public void buildClasifier(Instances instancias) {
		this.misinstancias=instancias;
		this.modelos = new Modelo[this.bagnum];
		Instances nuevas;
		for (int i = 0; i < modelos.length; i++) {
			nuevas = this.crearMuestras();
			modelos[i] = new NuestroModelo(2, DistanceWight.NoDistance, 1);
			modelos[i].buildClasifier(nuevas);
		}
	}

	@Override
	public void evaluarModelo(Instances instanciasAEvaluar) {
		Instance in;
		this.matrizConf = new int[instanciasAEvaluar.numClasses()][instanciasAEvaluar.numClasses()];
		for (int f = 0; f < this.matrizConf.length; f++) {
			for (int i = 0; i < this.matrizConf.length; i++) {
				this.matrizConf[f][i] = 0;
			}
		}
		for (int i = 0; i < instanciasAEvaluar.size(); i++) {
			in = instanciasAEvaluar.get(i);
			double predClass = this.clasificarInstancia(in);
			this.matrizConf[(int) predClass][(int)in.classValue()]++;	
		}
	}

	@Override
	public double clasificarInstancia(Instance NoClasificada) {
		double[] clases = new double[this.misinstancias.numClasses()];
		for (int j = 0; j < modelos.length; j++) { 
			try{
				clases[(int)modelos[j].clasificarInstancia(NoClasificada)]++;
			}catch(ArrayIndexOutOfBoundsException exception){
				
			}
			
		}
		double maxpos = 0;
		double max = 0;
		for (int i = 0; i < clases.length; i++) {
			if (max < clases[i]){
				max = clases[i];
				maxpos = i;
			}
		}
		return maxpos;
	}
}
