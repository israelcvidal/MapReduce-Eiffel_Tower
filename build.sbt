name := "MapReduce-Eiffel_Tower"

version := "1.0"

scalaVersion := "2.10.4"

val sparkVersion = "2.1.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies +=  "org.apache.spark" %% "spark-sql" % sparkVersion

libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.5.1"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.5.1" classifier "models"

lazy val gitRepo = "http://github.com/vspiewak/twitter-sentiment-analysis.git"

lazy val g = RootProject(uri(gitRepo))

lazy val root = project in file(".") dependsOn g in file(".") dependsOn g