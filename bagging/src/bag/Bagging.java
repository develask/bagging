package bag;

import src.Lector;
import weka.core.Instances;

public class Bagging {
	
	private int bagnum;
	private Instances misinstancias;
	
	public Bagging(int l,Instances ins) {
		bagnum = l;
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
		//TODO nuevo knn
		for (int i = 2; i <= 15; i++) {
			Instances boost = crearMuestras(instancias);
			Bagging nb= new Bagging(i, boost);
		}
		
	}
	
}
