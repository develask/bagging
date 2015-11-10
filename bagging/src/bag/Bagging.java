package bag;

import java.util.ArrayList;
import java.util.Enumeration;

import src.Lector;
import src.Modelo;
import src.NuestroModelo;
import src.NuestroModelo.DistanceWight;
import weka.core.Instance;
import weka.core.Instances;

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
		Instances ins = Lector.getLector().leerInstancias("./iris.arff");
		Instances[][] trozos = new Instances[10][2];
		int size =  ins.size()/10;
		double acumacc;
		double accuracyMax = 0.0;
		Bagging b;
		int numeroL = 0;
		for (int i = 2; i <= 2; i++) {
			acumacc = 0.0;
			b = new Bagging(i);
			//TODO
			for (int j = 1; j <= 1; j++) {
				trozos[j-1][0] = new Instances(ins, 0, size*(j-1));
				trozos[j-1][0].addAll(new Instances(ins, size*j, ins.size()-(size*j)));
				trozos[j-1][1] = new Instances(ins, size*(j-1), size);
			}
			//TODO
			for (int k = 0; k < 1; k++) {
				b.buildClasifier(trozos[k][0]);
				b.evaluarModelo(trozos[k][1]);
				System.out.println(b.calcularMediciones());
				acumacc+=b.accuracy();
			}
			//TODO
			acumacc=acumacc/1.0;
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
		System.out.println(correct + " - " + count + " | " + correct/count);
		return correct/count;
	}

	@Override
	public void buildClasifier(Instances instancias) {
		this.misinstancias=instancias;
		this.modelos = new Modelo[this.bagnum];
		Instances nuevas;
		for (int i = 0; i < modelos.length; i++) {
			nuevas = this.crearMuestras();
			modelos[i] = new NuestroModelo(135, DistanceWight.NoDistance, 1);
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
		//TODO
		for (int i = 0; i < 1 ; i++){// instanciasAEvaluar.size(); i++) {
			in = instanciasAEvaluar.get(i);
			double predClass = this.clasificarInstancia(in);
			this.matrizConf[(int) predClass][(int)in.classValue()]++;	
		}
	}

	@Override
	public double clasificarInstancia(Instance NoClasificada) {
		double[] clases = new double[this.misinstancias.numClasses()];
		//TODO
		for (int j = 0; j < 1; j++){// modelos.length; j++) { 
			int vari = (int)modelos[j].clasificarInstancia(NoClasificada);
			System.out.println(vari);
			clases[vari]++;
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
