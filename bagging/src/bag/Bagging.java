package bag;

import java.io.IOException;
import java.text.DecimalFormat;
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
		long TInicio, TFin, tiempo; //Variables para determinar el tiempo de ejecución
		TInicio = System.currentTimeMillis(); //Tomamos la hora en que inicio el algoritmo y la almacenamos en la variable inicio
		Instances ins1 = Lector.getLector().leerInstancias("./iris.arff");
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
		double accuracyMax = 0.00000;
		Bagging b;
		int numeroL = 0;
		for (int i = 2; i <= 15; i++) {
			long TIniciodentro=System.currentTimeMillis();;
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
				acumacc+=b.accuracy();
			}
			acumacc=(double)acumacc/10.0;
			System.out.println("L: "+i+" | Accuracy: "+acumacc);
			if(acumacc > accuracyMax){
				accuracyMax = acumacc;
				numeroL=i;
			}
			TFin = System.currentTimeMillis(); //Tomamos la hora en que finalizó el algoritmo y la almacenamos en la variable T
			tiempo = TFin - TIniciodentro; //Calculamos los milisegundos de diferencia
			System.out.println("Tiempo de ejecución en milisegundos: " + tiempo); //Mostramos en pantalla el tiempo de ejecución en milisegundos
		}
		System.out.println("\n\nBest L: "+numeroL+" | Accuracy: "+accuracyMax);
		TFin = System.currentTimeMillis(); //Tomamos la hora en que finalizó el algoritmo y la almacenamos en la variable T
		tiempo = TFin - TInicio; //Calculamos los milisegundos de diferencia
		System.out.println("Tiempo de ejecución en milisegundos: " + tiempo); //Mostramos en pantalla el tiempo de ejecución en milisegundos

	}

	@Override
	public String calcularMediciones() {
		String result = "";
		Enumeration<Object> clases=this.misinstancias.classAttribute().enumerateValues();
		result += ("------------Matriz De ConfusiÃ³n-------\n");
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
		return (double)correct/count;
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
