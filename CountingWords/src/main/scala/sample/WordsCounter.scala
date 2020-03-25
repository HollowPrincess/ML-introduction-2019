package sample
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


object WordsCounter {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().
      setMaster("local").
      setAppName("LearnScalaSpark")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val rddFromFile = sc.textFile("data/ml_part.txt")

    val res = rddFromFile.flatMap(line => line.split(" ")).map(word => word.toLowerCase)
    val res2 = res.map(x => (x, 1L)).reduceByKey(_ + _).sortBy(_._2,false)

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val df = spark.createDataFrame(res2).toDF("word", "amount")
    val res3 = df.limit(100)

    res3.write.format("csv").save("data/ml_part_res")
  }
}
