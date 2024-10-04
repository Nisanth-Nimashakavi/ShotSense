package com.shot.shotsense

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi

class BootReceiver : BroadcastReceiver() {
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onReceive(context: Context, intent: Intent?) {
        Log.d("test", "Received boot completed event")

        val startServiceIntent = Intent(context, MainActivity::class.java)
        context.startService(startServiceIntent)
    }
}
