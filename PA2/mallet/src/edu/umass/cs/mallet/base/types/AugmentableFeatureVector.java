/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */




/** 
		@author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
*/

package edu.umass.cs.mallet.base.types;

import edu.umass.cs.mallet.base.util.PropertyList;
import java.io.*;

public class AugmentableFeatureVector extends FeatureVector implements Serializable
{
	int size;															// max index with valid indices[] or values[] value 
	int maxSortedIndex;										/* if indices != null, top of values[] and indices[]
																					 may be unsorted indices. */

	// xxx Also make constructors for dense vectors, and add the appropriate
	// functionality in methods below.

	/** To make a binary vector, pass null for "values" */
	public AugmentableFeatureVector (Alphabet dict,
																	 int[] indices, double[] values, 
																	 int capacity, int size,
																	 boolean copy, boolean checkIndicesSorted,
																	 boolean removeDuplicates)
	{
		super (dict, indices, values, capacity, size, copy, checkIndicesSorted, removeDuplicates);
		this.size = size;
		this.maxSortedIndex = size - 1;
	}

	public AugmentableFeatureVector (Alphabet dict,
																	 int[] indices, double[] values, int capacity,	boolean copy,
																	 boolean checkIndicesSorted) {
		this (dict, indices, values, capacity, indices.length, copy, checkIndicesSorted, true);
	}
	public AugmentableFeatureVector (Alphabet dict,
																	 int[] indices, double[] values, int capacity,	boolean copy) {
		this (dict, indices, values, capacity, indices.length, copy, true, true); }
	public AugmentableFeatureVector (Alphabet dict,
																	 int[] indices, double[] values, int capacity) {
		this (dict, indices, values, capacity, indices.length, true, true, true); }

	public AugmentableFeatureVector (Alphabet dict, double[] values, int capacity) {
		this (dict, null, values, capacity, values.length, true, true, true); }
	public AugmentableFeatureVector (Alphabet dict, double[] values) {
		this (dict, null, values, values.length, values.length, true, true, true); }
	
	public AugmentableFeatureVector (Alphabet dict, int capacity, boolean binary) {
		// yyy
		this (dict, new int[capacity], binary ? null : new double[capacity],
					capacity, 0, false, false, false); }
	public AugmentableFeatureVector (Alphabet dict, boolean binary) {
		this (dict, 4, binary); }
	public AugmentableFeatureVector (Alphabet dict) {
		this (dict, false); }
	public AugmentableFeatureVector (FeatureVector fv) {
		this ((Alphabet)fv.dictionary, fv.indices, fv.values,
					fv.indices == null ? fv.values.length : fv.indices.length,
					fv.indices == null ? fv.values.length : fv.indices.length,
					true, false, false);
	}

	public AugmentableFeatureVector (FeatureSequence fs, boolean binary) {
		this (fs.getAlphabet(), binary);
		for (int i = fs.size()-1; i >= 0; i--)
			add (fs.getIndexAtPosition(i), 1.0);
	}

	public AugmentableFeatureVector (Alphabet dict, PropertyList pl, boolean binary,
																	 boolean growAlphabet) {
		this (dict, binary);
		if (pl == null)
			return;
		PropertyList.Iterator iter = pl.numericIterator();
		while (iter.hasNext()) {
			iter.nextProperty();
			//System.out.println ("AugmentableVector ("+dict.size()+") adding "+iter.getKey()+" "+iter.getNumericValue());
			int index = dict.lookupIndex (iter.getKey(), growAlphabet);
			if (index >= 0)
				add (index, iter.getNumericValue());
		}
	}

	public AugmentableFeatureVector (Alphabet dict, PropertyList pl, boolean binary) {
		this (dict, pl, binary, true);
	}
	
	// Aims to be cheap, constant time when (indices != null)
	public void add (int index, double value) {
		if (values == null && value != 1.0)
			throw new IllegalArgumentException ("Trying to add non-1.0 value to binary vector");
		assert (index >= 0);
		if (indices == null) {
			if (index >= values.length) {
				int newLength = index + 10;			// ???
				double[] newValues = new double[newLength]; // ???
				System.arraycopy (values, 0, newValues, 0, values.length);
				values = newValues;
				values[index] = value;
				assert (size <= index);
			} else {
				values[index] += value;
			}
			if (size <= index)
				size = index+1;
		} else {
			if (size == indices.length) {
				int newLength;
				if (indices.length == 0)
					newLength = 4;
				else if (indices.length < 4)
					newLength = indices.length * 2;
				else if (indices.length < 100)
					newLength = (indices.length * 3) / 2;
				else
					newLength = indices.length + 150;
				if (values != null) {
					double[] newValues = new double[newLength];
					System.arraycopy (values, 0, newValues, 0, values.length);
					values = newValues;
				}
				int[] newIndices = new int[newLength];
				System.arraycopy (indices, 0, newIndices, 0, indices.length);
				indices = newIndices;
			}
			//System.out.println ("indices.length="+indices.length+" size="+size);
			indices[size] = index;
			if (values != null)
				values[size] = value;
			size++;
		}
	}

	public void add (Object key, double value)
	{
		//System.out.println ("AugmentableFeatureVector dictionary = "+dictionary+", size = "+dictionary.size());
		int index = dictionary.lookupIndex (key);
		//System.out.println ("AugmentableFeatureVector index("+key+") = "+index);
		assert (index != -1);
		add (index, value);
	}
	
	public void add (int index) {
		if (values != null)
			throw new IllegalArgumentException ("Trying to add binary feature to real-valued vector");
		assert (index >= 0);
		
	}


	public final int numLocations () {
		if (indices == null)
			//return values.length;
			return size;
		if (size-1 != maxSortedIndex)
			sortIndices();
		return size;
	}

	public final int location (int index) {
		if (indices == null)
			return index;
		if (size-1 != maxSortedIndex)
			sortIndices();
		// Arrays.binarySearch (indices, index) doesn't work, because of the unused portion of the array at the end.
		for (int i = 0; i < size; i++) {
			if (indices[i] == index)
				return i;
			else if (indices[i] > index)
				return -1;
		}
		return -1;
	}

	public final double valueAtLocation (int location) {
		if (indices == null)
			return values[location];
		if (size-1 != maxSortedIndex)
			sortIndices();
		return super.valueAtLocation (location);
	}

	public final int indexAtLocation (int location) {
		if (indices == null)
			return location;
		if (size-1 != maxSortedIndex)
			sortIndices();
		assert (location < size);
		return super.indexAtLocation (location);
	}
	
	public final double value (int index) {
		if (indices == null)
			return values[index];
		if (size-1 != maxSortedIndex)
			sortIndices();
		int loc = location(index);
		if (loc >= 0) {
			if (values == null)
				return 1.0;
			else
				return values[loc];
		} else
			return 0;
	}

	public final void addTo (double[] accumulator, double scale)
	{
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		if (indices == null) {
			for (int i = 0; i < size; i++)
				accumulator[i] += values[i] * scale;
		} else if (values == null) {
			for (int i = 0; i < size; i++)
				accumulator[indices[i]] += scale;
		} else {
			for (int i = 0; i < size; i++)
				accumulator[indices[i]] += values[i] * scale;
		}
	}

	public final void addTo (double[] accumulator) {
		addTo (accumulator, 1.0);
	}

	public final void setValue (int index, double value) {
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		assert (values != null);
		if (indices == null) {
			assert (index < size);
			values[index] = value;
		} else {
			values[location(index)] = value;
		}
	}

	public final void setValueAtLocation (int location, double value) {
		assert (location < size);
		values[location] = value;
	}
	
	public ConstantMatrix cloneMatrix () {
		return new AugmentableFeatureVector ((Alphabet)dictionary,
																				 indices, values, indices.length, size,
																				 true, false, false);
	}

	public ConstantMatrix cloneMatrixZeroed () {
		if (indices == null)
			return new AugmentableFeatureVector (dictionary, new double[values.length]);
		else {
			int[] newIndices = new int[indices.length];
			System.arraycopy (indices, 0, newIndices, 0, indices.length);
			return new AugmentableFeatureVector (dictionary, newIndices, new double[values.length],
																					 values.length, values.length,
																					 false, false, false);
		}
	}
	
	public int singleSize () {
		return (indices == null
						? values.length
						: (size == 0
							 ? 0
							 : indices[size-1]));
	}
	
	public ConstantVector toConstantVector () {
		if (size-1 != maxSortedIndex)
			sortIndices();
		//System.out.println ("AugmentableFeatureVector toConstantVector size="+size);
		return new ConstantVector (indices, values, size, size, true, false, false);
	}

	public FeatureVector toFeatureVector () {
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		return new FeatureVector ((Alphabet)dictionary,
															indices, values, size, size, true, false, false);
	}
	
	public double dotProduct (DenseVector v) {
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		double ret = 0;
		if (values == null)
			for (int i = 0; i < size; i++)
				ret += v.value(indices[i]);
		else if (indices == null)
			for (int i = 0; i < size; i++)
				ret += values[i] * v.value(i);
		else
			for (int i = 0; i < size; i++)
				ret += values[i] * v.value(indices[i]);
		return ret;
	}

	public final double dotProduct (ConstantVector v) {
		if (v instanceof AugmentableFeatureVector)
			return dotProduct((AugmentableFeatureVector)v);
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		double ret = 0;
		int vl = 0;
		int vnl = v.numLocations ();
		if (values == null) {
			for (int i = 0; i < size; i++) {
				while (vl < vnl && v.indexAtLocation(vl) < indices[i])
					vl++;
				if (vl < vnl && v.indexAtLocation(vl) == indices[i])
					ret += v.valueAtLocation(vl);
			}
		} else if (indices == null) {
			for (int i = 0; i < vnl; i++) {
				int index = v.indexAtLocation(i);
				if (index < size)
					ret += v.valueAtLocation(i) * values[index];
			}
		} else {
			for (int i = 0; i < size; i++) {
				while (vl < vnl && v.indexAtLocation(vl) < indices[i])
					vl++;
				if (vl < vnl && v.indexAtLocation(vl) == indices[i])
					ret += values[i] * v.valueAtLocation(indices[vl]);
			}
		}
		return ret;
	}

	public final double dotProduct (AugmentableFeatureVector v) {
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		if (v.indices != null && v.size-1 != v.maxSortedIndex)
			v.sortIndices();
		double ret = 0;
		int vl = 0;
		int vnl = v.size;
		if (values == null) {
			if (v.values == null) {
				for (int i = 0; i < size; i++) {
					while (vl < vnl && v.indices[vl] < indices[i])
						vl++;
					if (vl < vnl && v.indices[vl] == indices[i])
						ret += 1.0;
				}
			} else {
				for (int i = 0; i < size; i++) {
					while (vl < vnl && v.indices[vl] < indices[i])
						vl++;
					if (vl < vnl && v.indices[vl] == indices[i])
						ret += v.values[vl];
				}
			}
		} else if (indices == null) {
			for (int i = 0; i < vnl; i++) {
				int index = v.indexAtLocation(i);
				if (index < size)
					ret += v.valueAtLocation(i) * values[index];
			}
		} else {
			if (v.values == null) {
				for (int i = 0; i < size; i++) {
					while (vl < vnl && v.indices[vl] < indices[i])
						vl++;
					if (vl < vnl && v.indices[vl] == indices[i])
						ret += values[i];
				}
			} else {
				for (int i = 0; i < size; i++) {
					while (vl < vnl && v.indices[vl] < indices[i])
						vl++;
					if (vl < vnl && v.indices[vl] == indices[i])
						ret += values[i] * v.values[vl];
				}
			}
		}
		return ret;
	}

	public void plusEquals (AugmentableFeatureVector v, double factor) {
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		if (v.indices != null && v.size-1 != v.maxSortedIndex)
			v.sortIndices();
		int vl = 0;
		int vnl = v.size;
		assert (values != null);
		if (indices == null) {
			if (v.indices == null) {
				vnl = Math.min (vnl, size);
				for (int i = 0; i < vnl; i++)
					values[i] += v.values[i];
			} else {
				// v.indices != null
				for (int i = 0; i < vnl; i++) {
					int index = v.indices[i];
					if (index < values.length) {
						values[index] += v.values[i] * factor;
						if (index >= size)
							size = index+1;
					}
				}
			}
		} else {
			// indices != null
			if (v.indices == null) {
				for (int i = 0; i < size; i++) {
					if (indices[i] < vnl)
						values[i] += v.values[indices[i]];
					// xxx We should check to see if there were more
					// higher indices in "v" that didn't get added!
				}
			} else {
				// v.indices != null
				if (v.values == null) {
					// v.indices != null && v.values == null
					for (int i = 0; i < size; i++) {
						while (vl < vnl && v.indices[vl] < indices[i])
							vl++;
						if (vl < vnl && v.indices[vl] == indices[i]) 
							values[i] += factor;
						// xxx We should check to see if there were more
						// higher indices in "v" that didn't get added!
					}
				} else {
					// v.indices != null && v.values != null
					for (int i = 0; i < size; i++) {
						while (vl < vnl && v.indices[vl] < indices[i])
							vl++;
						if (vl < vnl && v.indices[vl] == indices[i]) 
							values[i] += v.values[vl] * factor;
						// xxx We should check to see if there were more
						// higher indices in "v" that didn't get added!
					}
				}
			}
		}
	}

	// But only adds to those entries that have "locations" (i.e. are already non-zero)
	public void plusEquals (ConstantVector v, double factor) {
		if (v instanceof AugmentableFeatureVector) {
			plusEquals ((AugmentableFeatureVector)v, factor);
			return;
		}
		//assert (false) : v.getClass().getName(); // This code needs to be checked!
		if (indices != null && size-1 != maxSortedIndex)
			sortIndices();
		int vl = 0;
		assert (values != null);
		if (indices == null) {
			if (v.indices == null) {
				// indices == null && v.indices == null (&& v.values != null)
				int s = Math.min (size, v.values.length);
				for (int i = 0; i < s; i++)
					values[i] += v.values[i] * factor;
				// xxx We aren't adding in values with indices higher than "this.size"!
			} else {
				// indices == null && v.indices != null
				if (v.values == null) {
					// indices == null && v.indices != null && v.values == null
					for (int i = 0; i < v.indices.length; i++) {
						int index = v.indices[i];
						if (index < size)
							values[index] += factor;
					}
					// xxx We aren't adding in values with indices higher than "size"!
				} else {
					// indices == null && v.indices != null && v.values != null
					for (int i = 0; i < v.indices.length; i++) {
						int index = v.indices[i];
						if (index < size)
							values[index] += v.values[i] * factor;
						// xxx We aren't adding in values with indices higher than "size"!
					}
				}
			}
		} else {
			// indices != null
			if (v.indices == null) {
				// indices != null && v.indices == null (&& v.values != null)
				for (int i = 0; i < size; i++)
					if (indices[i] < v.values.length)
						values[i] += v.values[indices[i]] * factor;
				// xxx We aren't adding in values with indices higher than "size"!
			} else {
				// indices != null && v.indices != null
				int vnl = v.indices.length;
				if (v.values == null) {
					// indices != null && v.indices != null && v.values == null
					for (int i = 0; i < size; i++) {
						while (vl < vnl && v.indices[vl] < indices[i])
							vl++;
						if (vl < vnl && v.indices[vl] == indices[i]) 
							values[i] += v.values[vl] * factor;
						// xxx We should check to see if there were more
						// higher indices in "v" that didn't get added!
					}
				} else {
					// indices != null && v.indices != null && v.values != null
					for (int i = 0; i < size; i++) {
						while (vl < vnl && v.indices[vl] < indices[i])
							vl++;
						if (vl < vnl && v.indices[vl] == indices[i]) 
							values[i] += v.values[vl] * factor;
						// xxx We should check to see if there were more
						// higher indices in "v" that didn't get added!
					}
				}
			}
		}
	}

	public void plusEquals (ConstantVector v) {
		plusEquals (v, 1.0);
	}
	
	public void setAll (double v)
	{
		assert (values != null);
		for (int i = 0; i < values.length; i++)
			values[i] = v;
	}
	
	public double oneNorm () {
		if (size-1 != maxSortedIndex)
			sortIndices();
		double ret = 0;
		if (values == null)
			return size;
		for (int i = 0; i < size; i++)
			ret += values[i];
		return ret;
	}
	
	public double twoNorm () {
		if (size-1 != maxSortedIndex)
			sortIndices();
		double ret = 0;
		if (values == null)
			return Math.sqrt (size);
		for (int i = 0; i < size; i++)
			ret += values[i] * values[i];
		return Math.sqrt (ret);
	}
	
	public double infinityNorm () {
		if (size-1 != maxSortedIndex)
			sortIndices();
		if (values == null)
			return 1.0;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < size; i++)
			if (Math.abs(values[i]) > max)
				max = Math.abs(values[i]);
		return max;
	}		

	public void print() {
		//System.out.println ("ASV size="+size+" dict.size="+dictionary.size()+" values.length="+values.length+" indices.length="+indices.length);
		if (size-1 != maxSortedIndex)
			sortIndices();
		super.print();
	}

	
	protected void sortIndices ()
	{
		//System.out.println ("AFV sortIndices size="+size+" maxSortedIndex="+maxSortedIndex);
		if (indices == null)
			return;
		int numDuplicates = 0;
		// Just BubbleSort; this is efficient when already mostly sorted.
		// Note that we BubbleSort from the the end forward; this is most efficient
		//  when we have added a few additional items to the end of a previously sorted list.
		//  Note that we remember the highest index that was already sorted as "maxSortedIndex".
		// Note that maxSortedIndex may be -1 here, so the first time through the outer loop
		// just falls through immediately when the termination condition of the inner loop is met.
		for (int i = maxSortedIndex+1; i < size; i++) {
			//System.out.print ("fori="+i+"maxi="+maxSortedIndex+' ');
			boolean iIsDuplicate = false;
			for (int j = i; j > 0; j--) {
				//System.out.print ("i,j="+i+','+j+' ');
				if (indices[j] == indices[j-1]) {
					if (iIsDuplicate == false) {
						numDuplicates++;
						iIsDuplicate = true;
						//System.out.println (dictionary.lookupObject(indices[j]).toString()+" j="+j+" numDuplicates="+numDuplicates);
					}
				} else if (indices[j] < indices[j-1]) {
					// Swap both indices and values
					int f;
					f = indices[j];
					indices[j] = indices[j-1];
					indices[j-1] = f;
					if (values != null) {
						double v;
						v = values[j];
						values[j] = values[j-1];
						values[j-1] = v;
					}
				} else {
					break;
				}
			}
		}
		maxSortedIndex = size-1;
		if (numDuplicates > 0)
			removeDuplicates (numDuplicates);
	}
	
	// Argument zero is special value meaning that this function should count them.
	protected void removeDuplicates (int numDuplicates)
	{
		if (indices == null)
			return;
		//System.out.print ("AFV removeDuplicates ");
		//for (int i = 0; i < size; i++)
		//System.out.print (" " + dictionary.lookupObject(indices[i]) + "=" + indices[i]);
		//System.out.println (" numDuplicates="+numDuplicates);
		
		if (numDuplicates == 0)
			for (int i = 1; i < size; i++)
				if (indices[i-1] == indices[i])
					numDuplicates++;
		if (numDuplicates == 0)
			return;
		assert (indices.length - numDuplicates > 0)
			: "size="+size+" indices.length="+indices.length+" numDuplicates="+numDuplicates;
		int[] newIndices = new int[indices.length - numDuplicates];
		double[] newValues = values == null ? null : new double[indices.length - numDuplicates];
		newIndices[0] = indices[0];
		assert (indices.length >= size);
		for (int i = 1, j = 1; i < size; i++) {
			if (indices[i] == indices[i-1]) {
				if (values != null)
					newValues[j-1] += values[i];
			} else {
				newIndices[j] = indices[i];
				if (values != null)
					newValues[j] = values[i];
				j++;
			}
		}
		this.indices = newIndices;
		this.values = newValues;
		this.size -= numDuplicates;
	}

	// Serialization 
		
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;

	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);
		out.writeInt (size);
		out.writeInt (maxSortedIndex);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
		int version = in.readInt ();
		size = in.readInt();
		maxSortedIndex = in.readInt();
	}

}
