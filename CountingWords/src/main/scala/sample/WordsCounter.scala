package sample
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.immutable.ListMap

object WordsCounter {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().
      setMaster("local").
      setAppName("LearnScalaSpark")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val rddFromFile = sc.textFile("data/ml_part.txt")

    val res = rddFromFile.map(line => line.split(" ")).flatMap(word => word).map(word => word.toLowerCase).countByValue( ).toSeq.sortWith(_._2 > _._2).slice(0,100)
    val rddRes = sc.parallelize(res.toSeq)
    rddRes.coalesce(1).saveAsTextFile("data/ml_part_res")
  }
}
