package bag;

import java.util.ArrayList;

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
		double acumacc=0.0;
		double accuracyMax = 0.0;
		Bagging b;
		int numeroL = 0;
		for (int i = 2; i <= 15; i++) {
			b = new Bagging(i);
			for (int j = 1; j <= 10; j++) {
				trozos[j-1][0] = new Instances(new Instances(ins, 0, size*(j-1)));
				trozos[j-1][0].addAll(new Instances(ins, size*j, ins.size()));
				trozos[j-1][1] = new Instances(ins, size*(j-1), size*j);
			}
			for (int k = 0; k < 10; k++) {
				b.buildClasifier(trozos[k][0]);
				b.evaluarModelo(trozos[k][1]);
				acumacc+=b.accuracy();
			}
			acumacc=acumacc/10;
			if(acumacc > accuracyMax){
				accuracyMax = acumacc;
				numeroL=i;
			}
		}
		
		

		
	}

	@Override
	public String calcularMediciones() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double accuracy() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void buildClasifier(Instances instancias) {
		this.misinstancias=instancias;
		
	}

	@Override
	public void evaluarModelo(Instances instanciasAEvaluar) {
		for (int i = 0; i < instanciasAEvaluar.size(); i++) { 
			double[] clases = new double[instanciasAEvaluar.numClasses()];
			int clase = 
			
		}
	}

	@Override
	public double clasificarInstancia(Instance NoClasificada) {
		Modelo[] modelos = new Modelo[this.bagnum];
		Instances nuevas;
		for (int i = 0; i < modelos.length; i++) {
			nuevas = this.crearMuestras();
			modelos[i] = new NuestroModelo(2, DistanceWight.NoDistance, 1);
			modelos[i].buildClasifier(nuevas);
		}
		double[] clases = new double[this.misinstancias.numClasses()];
		for (int j = 0; j < modelos.length; j++) {
			clases[(int)modelos[j].clasificarInstancia(NoClasificada)]++;
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
