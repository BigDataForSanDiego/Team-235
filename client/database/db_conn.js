var mysql = require('mysql');
var connection = mysql.createConnection({
    host: 'localhost',
    user: 'spare.use673@gmail.com',
    password: 'AlgoTech'
})

connection.connect(function(err){
    if (err) {
        console.error('error connecting: ' + err.stack);
        return;
      }
    
      console.log('connected as id ' + connection.threadId);
});