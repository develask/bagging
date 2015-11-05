package bag;

import java.util.ArrayList;

import src.Lector;
import src.NuestroModelo;
import src.NuestroModelo.DistanceWight;
import weka.core.Instance;
import weka.core.Instances;

public class Bagging {
	
	private int bagnum;
	private Instances misinstancias;
	
	public Bagging(int l,Instances ins) {
		misinstancias = ins;
	}
	
	private static Instances crearMuestras(Instances is){
		Instances instancias= new Instances(is,0);
		for (int i = 0; i < is.numInstances(); i++) {
			int random = (int)Math.floor(Math.random()*is.numInstances()); 
			instancias.add(is.get(random));
		}
		return instancias;
	}
	
	public static void main(String[] args) {
		Instances instancias = Lector.getLector().leerInstancias("./iris.arff");
		NuestroModelo noum= new NuestroModelo(2, DistanceWight.NoDistance, 1);
		Instance instanciaAClasificar = Lector.getLector().leerInstancias("Como vendria la instancia?'?'");
		double accuracyMax = 0.0,accuracy=0.0;
		ArrayList<Integer> valVot = new ArrayList<Integer>(instancias.numClasses());
		int numeroL = 0;
		for (int i = 2; i <= 15; i++) {
			Instances boost = crearMuestras(instancias);
			noum.buildClasifier(boost);
			noum.evaluarModelo();
			accuracy = noum.accuracy();
			if(accuracy > accuracyMax){
				accuracyMax = accuracy;
				numeroL=i;
			}
			double valorClase = noum.clasificarInstancia(instanciaAClasificar);
			//falta coger el indice por el que la instancia se ha clasificado
			valVot.set(/*El valor de arriba*/, valVot.get(/*El valor de arriba*/));
		}

		
	}
	
}
