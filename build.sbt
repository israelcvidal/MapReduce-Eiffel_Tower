name := "MapReduce-Eiffel_Tower"

version := "1.0"

scalaVersion := "2.10.4"

val sparkVersion = "2.1.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies +=  "org.apache.spark" %% "spark-sql" % sparkVersion

libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.5.1"
