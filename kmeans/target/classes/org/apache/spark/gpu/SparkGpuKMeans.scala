package org.apache.spark.gpu

//import breeze.linalg.squaredDistance
import breeze.linalg.{Vector, DenseVector, squaredDistance}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import java.nio._
import java.util._

import org.jocl.CL._
import org.jocl._

object SparkGpuKMeans {	
		
	val err = Array.ofDim[Int](1)

	private val programSource =

	"""#pragma OPENCL EXTENSION cl_khr_fp64: enable
	  |#define DOUBLE_MAX 1.0e+50
	  |
	  |__kernel void kmeans(__global double *point, __global double *cluster, __global int *family, int point_num, int point_dim, int cluster_num) {
	  |		int point_id = get_global_id(0);
	  |		if (point_id < point_num) {
	  |			double min_dist = DOUBLE_MAX;
	  |			double tmp_dist;
	  |			int index = 0;
	  |			int	i, j;
	  |			for (i = 0; i < cluster_num; i++) {
	  |				tmp_dist = 0;
	  |				for (j = 0; j < point_dim; j++) {
	  |					tmp_dist += (point[point_id * point_dim + j] - cluster[i * point_dim + j]) * 
	  |						(point[point_id * point_dim + j] - cluster[i * point_dim + j]);
	  |				}
	  |				if (tmp_dist < min_dist) {
	  |					index = i;
	  |					min_dist = tmp_dist;
	  |				}
	  |			}		
	  |			family[point_id] = index; 
	  |		}
	  |	}""".stripMargin
  
	def parseVector(line: String): Vector[Double] = {
		DenseVector(line.split(' ').map(_.toDouble))
	 }

	private def getMaxWorkGroupSize(device: cl_device_id, paramName: Int): Int = {
		val size = Array(0L)
		clGetDeviceInfo(device, paramName, 0, null, size)

		val maxWorkGroupSize = Array.ofDim[Int](1) //Max allowed work-items in a group 
		clGetDeviceInfo(device, paramName, Sizeof.size_t, Pointer.to(maxWorkGroupSize), null)
		println("maxWorkGroupSize:" + maxWorkGroupSize(0))
		maxWorkGroupSize(0)
	}
	
	private def roundWorkSizeUp(groupWorkSize: Long, globalWorkSize: Long): Long = {
		val remainder = globalWorkSize % groupWorkSize
		if (remainder == 0) globalWorkSize else globalWorkSize + groupWorkSize - remainder
	}
	
	private def getString(device: cl_device_id, paramName: Int): String = {
		// Obtain the length of the string that will be queried
		val size = Array(0L)
		clGetDeviceInfo(device, paramName, 0, null, size)

		// Create a buffer of the appropriate size and fill it with the info
		val buffer = Array.ofDim[Byte](size(0).toInt)
		clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null)

		// Create a string from the buffer (excluding the trailing \0 byte)
		new String(buffer, 0, buffer.length - 1)
	}
    
	private def getTime(event : cl_event, commandType: Int): Double = {
		val array = Array.ofDim[Long](1)
        clGetEventProfilingInfo(event, commandType, Sizeof.cl_ulong, 
			Pointer.to(array), null)
        array(0) / 1000000.0
    }
	
	def initializationGpu() = {

		setExceptionsEnabled(true)
		// The platform, device type and device number
		// that will be used
		val platformIndex = 0
		val deviceType = CL_DEVICE_TYPE_ALL
		val deviceIndex = 0

		val platform = {
			// Obtain the number of platforms
			val numPlatformsArray = Array(0)
			clGetPlatformIDs(0, null, numPlatformsArray)
			val numPlatforms = numPlatformsArray(0)

			// Obtain a platform ID
			val platforms = Array.ofDim[cl_platform_id](numPlatforms)
			clGetPlatformIDs(platforms.length, platforms, null)
			platforms(platformIndex)
		}
		
		// Initialize the context properties
		val contextProperties = new cl_context_properties()
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform)
		
		val device = {
			// Obtain the number of devices for the platform
			val numDevicesArray = Array(0)
			clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
			val numDevices = numDevicesArray(0)

			// Obtain a device ID
			val devices = Array.ofDim[cl_device_id](numDevices)
			clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
			devices(deviceIndex)
		}
		
		// Create a context for the selected device
		val context = clCreateContext(contextProperties, 1, Array(device),
			null, null, null)
		
		val deviceName = getString(device, CL_DEVICE_NAME) // get the value of key: device
		println(s"CL_DEVICE_NAME: $deviceName")
		val maxWorkGroupSize = getMaxWorkGroupSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE)
		
		// Create a command-queue
		val properties = CL_QUEUE_PROFILING_ENABLE
		val commandQueue = clCreateCommandQueue(context, device, properties, err)

		// Create the program from the source code
		val program = clCreateProgramWithSource(context, 1, Array(programSource),
			null, err)

		// Build the program
		err(0) = clBuildProgram(program, 0, null, null, null, null)

		// Create the kernel
		val kernel = clCreateKernel(program, "kmeans", err)

		(context, commandQueue, kernel, program, maxWorkGroupSize)		
	}
	
	def main(args: Array[String]) {
		
		if (args.length < 3) {
			System.err.println("Usage: SparkGpuKMeans <file> <k> <convergeDist>")
			System.exit(1)
		}
		
		val sparkConf = new SparkConf().setAppName("SparkGpuKMeans")
		val sc = new SparkContext(sparkConf)
		
		val lines = sc.textFile(args(0), 1) //create RDD from external file
		val K = args(1).toInt
		val convergeDist = args(2).toDouble
		
		val pointsNum = lines.count().toInt     
		val data = lines.map(parseVector _).cache()
		
		var collectTime : Double= System.nanoTime()
		val pointsArray = data.collect().toArray
		collectTime = System.nanoTime() - collectTime

		val dim = pointsArray(0).length.toInt // the dimension of points
		
		val pointsFamilyArray = Array.ofDim[Int](pointsNum) // the location id and family id of points  
		val centerIndex = Array.ofDim[Int](K)
		for (i <- 0 until K)
			centerIndex(i) = i

		//val kPointsArray = data.takeSample(withReplacement = false, K, 42).toArray 
		val kPointsArray = data.take(K).toArray //cluster points 
		
		setExceptionsEnabled(true)
		val (context, commandQueue, kernel, program, maxWorkGroupSize) = initializationGpu()
		
		val globalWorkSize = Array.ofDim[Long](1)
		val localWorkSize = Array.ofDim[Long](1)
		localWorkSize(0) = maxWorkGroupSize
	
		globalWorkSize(0) = roundWorkSizeUp(localWorkSize(0), pointsNum)

		/* 
		* Creates a new Pointer to the given values or buffer 
		* oh,my god, Pointer cannot point to 2 dimensions array 
		*/
		
		var pointsArray1D = Array.ofDim[Double](pointsNum * dim)
		var kPointsArray1D = Array.ofDim[Double](K * dim)
		
		var transformTime = .0
		var transformStartTime = .0
		var transformEndTime = .0
		
		transformStartTime = System.nanoTime()
		for	(i <- 0 until pointsNum) {
			for(j <- 0 until dim)
				pointsArray1D(i * dim + j) = pointsArray(i)(j)
		}
		transformEndTime = System.nanoTime()
		transformTime += transformEndTime - transformStartTime

		/* Allocate the memory objects for the input- and output data */
		var pointsMem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
			Sizeof.cl_double * pointsNum * dim, null, err)
		var kPointsMem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
			Sizeof.cl_double * K * dim, null, err)
		var pointsFamilyMem = clCreateBuffer(context, CL_MEM_READ_WRITE, 
			Sizeof.cl_int * pointsNum, null, err)
		
		var pointsNumMem = Array.ofDim[Int](1)		
		var dimMem = Array.ofDim[Int](1)		
		var KMem = Array.ofDim[Int](1)		
		pointsNumMem(0) = pointsNum
		dimMem(0) = dim
		KMem(0) = K
		println()
		err(0) = clEnqueueWriteBuffer(commandQueue, pointsMem, CL_TRUE, 0, 
			Sizeof.cl_double * pointsNum * dim, Pointer.to(pointsArray1D), 0, null, null)
		
		var tempDist = Double.PositiveInfinity
		
		val kMeansStartTime = System.nanoTime()
		
		var kMeansKernelTime = .0
		var kMeansKernelStartTime = .0
		var kMeansKernelEndTime = .0
			
		var zipTime = .0
		var zipStartTime = .0 
		var zipEndTime = .0 
		
		var newPointsTime = .0
		var newPointsStartTime = .0 
		var newPointsEndTime = .0 
		
		var mapReduceTime = .0
		var mapReduceStartTime = .0 
		var mapReduceEndTime = .0 
		
		var insertTime = .0
		var insertStartTime = .0 
		var insertEndTime = .0 
		
		var kernelEvent = new cl_event()
		var readEvent = new cl_event()	
		var writeEvent = new cl_event()	
		
		while (tempDist > convergeDist) {
			
			transformStartTime = System.nanoTime()
			for	(i <- 0 until K) { // convert 2DArray to 1DArray
				for(j <- 0 until dim)
					kPointsArray1D(i * dim + j) = kPointsArray(i)(j)
			}
			transformEndTime = System.nanoTime()
			transformTime += transformEndTime - transformStartTime

			err(0) = clEnqueueWriteBuffer(commandQueue, kPointsMem, CL_TRUE, 0, 
				Sizeof.cl_double * K * dim, Pointer.to(kPointsArray1D), 0, null, writeEvent)
			err(0) = clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(pointsMem))
			err(0) |= clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(kPointsMem))
			err(0) |= clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(pointsFamilyMem))
			err(0) |= clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(pointsNumMem))
			err(0) |= clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(dimMem))
			err(0) |= clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(KMem))
			
			err(0) = clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, 
				globalWorkSize, localWorkSize, 0, null, kernelEvent)
			err(0) = clEnqueueReadBuffer(commandQueue, pointsFamilyMem, CL_TRUE, 0, 
				pointsNum * Sizeof.cl_int, Pointer.to(pointsFamilyArray), 0, null, readEvent)
			
			//clWaitForEvents(3, Array(writeEvent, kernelEvent, readEvent))
			clFinish(commandQueue)
			
			kMeansKernelStartTime = getTime(kernelEvent, CL_PROFILING_COMMAND_START)
			kMeansKernelEndTime = getTime(kernelEvent, CL_PROFILING_COMMAND_END)
			kMeansKernelTime += kMeansKernelEndTime - kMeansKernelStartTime
			zipStartTime = System.nanoTime()	
			val pointsFamilyRdd = sc.parallelize(pointsFamilyArray, 1)
			val zipRdd = pointsFamilyRdd.zip(data)
			zipEndTime = System.nanoTime()
			zipTime += zipEndTime - zipStartTime

			mapReduceStartTime = System.nanoTime()
			val closest = zipRdd.map (p => (p._1, (p._2, 1))) // be careful!!
			val pointsStats = closest.reduceByKey{case((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}
			mapReduceEndTime = System.nanoTime()
			mapReduceTime += mapReduceEndTime - mapReduceStartTime
			
			/*
			 *val average = pointsStats.map {pair => 
			 *	(pair._1, pair._2._1 * (1.0 / pair._2._2))}.sortByKey.collect.toBuffer
			*/
			newPointsStartTime = System.nanoTime()
			val average = pointsStats.map {pair => 
				(pair._1, pair._2._1 * (1.0 / pair._2._2))}.collectAsMap()// action: collect and then map
			newPointsEndTime = System.nanoTime()
			newPointsTime += newPointsEndTime - newPointsStartTime
			val newPoints = centerIndex.map(index => { 
				average.get(index) match {
					case Some(newCenter) => (index, newCenter)
					case None => (index, kPointsArray(index))
				}
			})
			
			tempDist = .0
			for (i <- 0 until K)
				tempDist += squaredDistance(kPointsArray(i), newPoints(i)._2)		
			
			/**
			 *insertStartTime = System.nanoTime()
			 *for (i <- 0 until K) {
			 *	if (newPoints.length < i + 1)
			 *		newPoints.insert(i, (i, kPointsArray(i)))
			 *	else if (i != newPoints(i)._1)
			 *		newPoints.insert(i, (i, kPointsArray(i)))
			 *	else
			 *		tempDist += squaredDistance(kPointsArray(i), newPoints(i)._2)		
			 *}
			 *insertEndTime = System.nanoTime()
			 *insertTime += insertEndTime - insertStartTime
			*/
			
			for (newP <- newPoints) {
				kPointsArray(newP._1) = newP._2
			}
			println("Finished iteration (delta = " + tempDist + ")")
		}
		
		println("Final centers:")
			kPointsArray.foreach(println)
		
		val kMeansFinishTime = System.nanoTime()
		val kMeansTime = (kMeansFinishTime - kMeansStartTime) / 1000000.
		collectTime /= 1000000.
		transformTime /= 1000000.
		zipTime /= 1000000.
		mapReduceTime /= 1000000.
		newPointsTime /= 1000000.
		//insertTime /= 1000000.	
		println("kMeansTime:" + kMeansTime + "ms")
		println("kMeansKernelTime:" + kMeansKernelTime + "ms")
		println("transformTime:" + transformTime + "ms")
		println("zipTime:" + zipTime + "ms")
		println("mapReduceTime:" + mapReduceTime + "ms")
		println("newPointsTime:" + newPointsTime + "ms")
		println("collectTime:" + collectTime + "ms")
		
		err(0) = clReleaseKernel(kernel)
		err(0) |= clReleaseProgram(program)
		err(0) |= clReleaseCommandQueue(commandQueue)
		err(0) |= clReleaseContext(context)
		err(0) |= clReleaseMemObject(pointsMem)
		err(0) |= clReleaseMemObject(kPointsMem)
		err(0) |= clReleaseMemObject(pointsFamilyMem)
	
		sc.stop()
	}

}
