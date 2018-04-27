package jvm

import nn.SampleNN

fun main(arguments: Array<String>) {

    useNN()
}

fun useNN(){
    val myClass = SampleNN()
    myClass.run()
}