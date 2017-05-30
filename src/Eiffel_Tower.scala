import org.apache.spark.{SparkConf}
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

/**
  * Created by israelcvidal on 28/05/17.
  */
object Eiffel_Tower {

    def main(args: Array[String]) {
        val k = 20

        Logger.getLogger("org").setLevel(Level.OFF)
        Logger.getLogger("akka").setLevel(Level.OFF)

        val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
        val spark = SparkSession
          .builder()
          .config(conf)
          .getOrCreate()


        val stopWords = spark.sparkContext.textFile("Dataset/stop-word-list.csv").flatMap(line => line.split(", ")).collect()
        val file = "Dataset/eiffel-tower-reviews.json"
        val textFile = spark.read.json(file).toDF

        val wordCount = textFile.select("text").rdd
            .flatMap(line => line.toString().split(" "))
            .map(word => word.toLowerCase)
            .filter(word => !stopWords.contains(word))
            .map(word => (word, 1))
            .reduceByKey(_ + _)

        val toRemove = "[]".toSet
        val sentenceCount = textFile.select("text").rdd
//          .flatMap(line => line.toString().split(" and ").)
          .flatMap(line => line.toString().split("\\.[^.]+"))
//          .flatMap(line => line.toString().split(" it "))
          .map(word => word.filterNot(toRemove))
          .map(word => word.toLowerCase)
          .map(word => (word, 1))
          .reduceByKey(_ + _)
//          .reduce(_2 )
        val topKwords = wordCount
          .sortBy(word => word._2, ascending = false)
          .take(k)

        val topKsentences = sentenceCount
          .sortBy(word => word._2, ascending = false)
          .take(k)
//        println("Top " + k + " words: ")
//        topKwords.foreach(println)

        println("Top " + k + " sentences: ")
        topKsentences.foreach(println)
    }
}
